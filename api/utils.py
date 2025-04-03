import logging
import logging.config
import yaml
import os
import re
from urllib.parse import urlparse, urlunparse
import httpx # Keep for URL pinging etc.

# --- Custom Exceptions ---
class RateLimitException(Exception):
    """Custom exception for downstream API rate limits."""
    pass

class ApiException(Exception):
    """Custom exception for general downstream API errors."""
    pass


# --- Config Loading ---
_config = None
def get_config():
    global _config
    if _config is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                _config = yaml.safe_load(f)
            if _config is None: _config = {} # Handle empty file
            # TODO: Add validation using Pydantic models for config?
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}. Using empty config.")
            _config = {}
        except yaml.YAMLError as e:
             logging.error(f"Error parsing configuration file {config_path}: {e}. Using empty config.")
             _config = {}
    return _config

# --- Logging Setup ---
_logging_configured = False
def setup_logging():
    global _logging_configured
    if _logging_configured: return

    config = get_config()
    log_level = config.get('logging', {}).get('level', 'INFO').upper()
    log_file_path = config.get('logging', {}).get('file_path', os.path.join('logs', 'api.log')) # Correct path handling

    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'level': log_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'filename': log_file_path,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False,
            },
            'uvicorn.error': { # Capture uvicorn errors
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False,
            },
             'uvicorn.access': { # Less verbose access logs maybe
                'handlers': ['console', 'file'],
                'level': 'WARNING', # Set to INFO for detailed access logs
                'propagate': False,
            },
            'httpx': { # Control httpx logging level
                'handlers': ['console', 'file'],
                'level': 'WARNING', # Set lower for debugging HTTP calls
                'propagate': False,
            },
             'multipart.multipart': { # Silence noisy multipart logs
                'handlers': ['console', 'file'],
                'level': 'WARNING',
                'propagate': False,
             }
             # Add specific loggers for langchain, transformers etc. if needed
        }
    }
    try:
         logging.config.dictConfig(logging_config)
         _logging_configured = True
         logging.info("Logging configured successfully.")
         # Quieten overly verbose libraries after initial setup
         logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
         logging.getLogger("multipart.multipart").setLevel(logging.WARNING) # Already set above, double ensure
    except Exception as e:
         # Fallback to basic config if dictConfig fails
         logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         logging.error(f"Failed to configure logging from dictConfig: {e}. Using basicConfig.")
         _logging_configured = True # Mark as configured anyway


# --- URL Utilities ---
def is_valid_url(url: str) -> bool:
    """Checks if a string is a potentially valid HTTP/HTTPS URL structure."""
    if not isinstance(url, str): return False
    try:
        result = urlparse(url)
        # Check for scheme (http/https) and netloc (domain name)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False


def sanitize_url_for_scan(url: str) -> str:
     """Prepares URL for sending to scanner APIs (removes fragments, standardizes)."""
     try:
          parsed = urlparse(url)
          # Remove fragment, potentially UTM parameters if desired
          parsed = parsed._replace(fragment="")
          # Example: remove common tracking params (add more as needed)
          query_params = [p for p in parsed.query.split('&') if not p.lower().startswith(('utm_', 'fbclid=', 'gclid='))]
          parsed = parsed._replace(query="&".join(query_params))

          # Rebuild and ensure http/https exists if missing (though is_valid_url should catch this)
          clean_url = urlunparse(parsed)
          if not clean_url.startswith(('http://', 'https://')):
               # Guess https, maybe fallback to http? or raise error?
               if url.startswith('//'): # Handle protocol-relative URLs
                   clean_url = 'https:' + clean_url
               elif urlparse(f"https://{url}").netloc: # Check if adding https makes it valid
                     clean_url= f"https://{url}"
               else:
                    # Maybe raise ValueError if it can't be reasonably fixed
                    logging.warning(f"Could not confidently add scheme to URL: {url}")
                    return url # Return original if unsure how to fix


          # Remove trailing slashes for consistency, unless it's just the domain
          if clean_url.endswith('/') and urlparse(clean_url).path != '/':
                clean_url = clean_url.rstrip('/')

          return clean_url
     except Exception as e:
          logging.error(f"Error sanitizing URL '{url}': {e}")
          return url # Return original on error


# Example ping_url if needed elsewhere (often not needed if scanners are used)
async def ping_url(url: str, timeout: int = 5) -> bool:
     """Checks if a URL is reachable with a HEAD request."""
     if not is_valid_url(url):
          return False
     try:
          async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
               response = await client.head(url)
               # Consider any success/redirect (2xx/3xx) as reachable
               return response.status_code < 400
     except (httpx.RequestError, httpx.TimeoutException) as e:
          logging.debug(f"Ping failed for {url}: {e}")
          return False
     except Exception as e:
         logging.warning(f"Unexpected error pinging {url}: {e}")
         return False

