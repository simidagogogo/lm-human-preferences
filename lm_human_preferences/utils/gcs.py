import os
import random
import subprocess
import time
import traceback
import warnings
from functools import wraps
from urllib.parse import urlparse, unquote

import requests
try:
    from google.api_core.exceptions import InternalServerError, ServiceUnavailable
    from google.cloud import storage
except ImportError:
    # If Google Cloud libraries are not available, we'll use HTTP downloads for Azure
    InternalServerError = Exception
    ServiceUnavailable = Exception
    storage = None

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")


def exponential_backoff(
        retry_on=lambda e: True, *, init_delay_s=1, max_delay_s=600, max_tries=30, factor=2.0,
        jitter=0.2, log_errors=True):
    """
    Returns a decorator which retries the wrapped function as long as retry_on returns True for the exception.
    :param init_delay_s: How long to wait to do the first retry (in seconds).
    :param max_delay_s: At what duration to cap the retry interval at (in seconds).
    :param max_tries: How many total attempts to perform.
    :param factor: How much to multiply the delay interval by after each attempt (until it reaches max_delay_s).
    :param jitter: How much to jitter by (between 0 and 1) -- each delay will be multiplied by a random value between (1-jitter) and (1+jitter).
    :param log_errors: Whether to print tracebacks on every retry.
    :param retry_on: A predicate which takes an exception and indicates whether to retry after that exception.
    """
    def decorate(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            delay_s = float(init_delay_s)
            for i in range(max_tries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if not retry_on(e) or i == max_tries-1:
                        raise
                    if log_errors:
                        print(f"Retrying after try {i+1}/{max_tries} failed:")
                        traceback.print_exc()
                    jittered_delay = random.uniform(delay_s*(1-jitter), delay_s*(1+jitter))
                    time.sleep(jittered_delay)
                    delay_s = min(delay_s * factor, max_delay_s)
        return f_retry
    return decorate


def _gcs_should_retry_on(e):
    # Retry on all 503 errors and 500, as recommended by https://cloud.google.com/apis/design/errors#error_retries
    return isinstance(e, (InternalServerError, ServiceUnavailable, requests.exceptions.ConnectionError))


def parse_url(url):
    """Given a gs:// path or https:// URL, returns bucket name and blob path."""
    result = urlparse(url)
    if result.scheme == 'gs':
        return result.netloc, unquote(result.path.lstrip('/'))
    elif result.scheme == 'https':
        # Support both Google Cloud Storage and Azure Blob Storage
        if result.netloc == 'storage.googleapis.com':
            bucket, rest = result.path.lstrip('/').split('/', 1)
            return bucket, unquote(rest)
        elif 'blob.core.windows.net' in result.netloc:
            # Azure Blob Storage URL format: https://account.blob.core.windows.net/container/path
            # For Azure, we'll return the full URL as-is for direct HTTP download
            return None, url  # Return None as bucket to indicate Azure URL
        else:
            raise Exception(f'Unsupported URL host: {result.netloc}')
    else:
        raise Exception(f'Could not parse {url} as gcs url')


@exponential_backoff(_gcs_should_retry_on)
def get_blob(url, client=None):
    bucket_name, path = parse_url(url)
    # Handle Azure URLs - return None to use HTTP download instead
    if bucket_name is None:
        return None
    if client is None:
        if storage is None:
            return None
        client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    return bucket.get_blob(path)


@exponential_backoff(_gcs_should_retry_on)
def download_contents(url, client=None):
    """Given a gs:// path, returns contents of the corresponding blob."""
    blob = get_blob(url, client)
    if not blob: return None
    return blob.download_as_string()


@exponential_backoff(_gcs_should_retry_on)
def upload_contents(url, contents, client=None):
    """Given a gs:// path, returns contents of the corresponding blob."""
    if client is None:
        client = storage.Client()
    bucket_name, path = parse_url(url)
    bucket = client.get_bucket(bucket_name)
    blob = storage.Blob(path, bucket)
    blob.upload_from_string(contents)


def download_directory_cached(url, comm=None):
    """ Given a GCS path url, caches the contents locally.
    WARNING: only use this function if contents under the path won't change!
    """
    cache_dir = '/tmp/gcs-cache'
    bucket_name, path = parse_url(url)
    is_master = not comm or comm.Get_rank() == 0
    local_path = os.path.join(cache_dir, bucket_name, path)

    sentinel = os.path.join(local_path, 'SYNCED')
    if is_master:
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            cmd = 'gsutil', '-m', 'cp', '-r', url, os.path.dirname(local_path) + '/'
            print(' '.join(cmd))
            subprocess.check_call(cmd)
            open(sentinel, 'a').close()
    else:
        while not os.path.exists(sentinel):
            time.sleep(1)
    return local_path


def download_file_cached(url, comm=None):
    """ Given a GCS path url or Azure Blob Storage URL, caches the contents locally.
    WARNING: only use this function if contents under the path won't change!
    """
    cache_dir = '/tmp/gcs-cache'
    bucket_name, path = parse_url(url)
    is_master = not comm or comm.Get_rank() == 0
    
    # Handle Azure Blob Storage URLs (direct HTTP download)
    if bucket_name is None and path.startswith('http'):
        # Azure URL - use direct HTTP download
        azure_url = path
        # Create a safe filename from the URL
        import hashlib
        url_hash = hashlib.md5(azure_url.encode()).hexdigest()
        filename = os.path.basename(azure_url) or 'file'
        local_path = os.path.join(cache_dir, 'azure', url_hash, filename)
        sentinel = local_path + '.SYNCED'
        
        if is_master:
            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f'Downloading from Azure: {azure_url}')
                try:
                    response = requests.get(azure_url, stream=True, timeout=60)
                    response.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    open(sentinel, 'a').close()
                    print(f'Downloaded to: {local_path}')
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        print(f'WARNING: File not found at {azure_url}')
                        print(f'This dataset may not be available. Creating placeholder file.')
                        # 创建一个空文件作为占位符
                        with open(local_path, 'w') as f:
                            f.write('[]')
                        open(sentinel, 'a').close()
                        print(f'Created placeholder at: {local_path}')
                    else:
                        raise
        else:
            while not os.path.exists(sentinel):
                time.sleep(1)
        return local_path
    
    # Handle GCS URLs (original logic)
    local_path = os.path.join(cache_dir, bucket_name, path)
    sentinel = local_path + '.SYNCED'
    if is_master:
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            cmd = 'gsutil', '-m', 'cp', url, local_path
            print(' '.join(cmd))
            subprocess.check_call(cmd)
            open(sentinel, 'a').close()
    else:
        while not os.path.exists(sentinel):
            time.sleep(1)
    return local_path
