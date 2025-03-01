import os
from openavmkit.cloud.azure import init_service_azure, get_creds_from_env_azure
from dotenv import load_dotenv

from openavmkit.cloud.base import CloudService, CloudType, CloudAccess, CloudCredentials
from openavmkit.cloud.huggingface import get_creds_from_env_huggingface, init_service_huggingface
from openavmkit.cloud.sftp import get_creds_from_env_sftp, init_service_sftp


def init(verbose: bool) -> CloudService | None:
  load_dotenv()
  cloud_type = os.getenv("CLOUD_TYPE")
  cloud_access = os.getenv("CLOUD_ACCESS")
  if cloud_type is None:
    raise ValueError("Missing 'CLOUD_TYPE' in environment. Have you created your .env file and properly filled it out?")
  if cloud_access is None:
    raise ValueError("Missing 'CLOUD_ACCESS' in environment. Have you created your .env file and properly filled it out?")
  cloud_type = cloud_type.lower()
  cloud_access = cloud_access.lower()
  credentials = _get_creds_from_env()

  try:
    cloud_service = _init_service(cloud_type, cloud_access, credentials)
  except ValueError as e:
    return None
  cloud_service.verbose = verbose
  return cloud_service


def _get_creds_from_env() -> CloudCredentials:
  cloud_type = os.getenv("CLOUD_TYPE").lower()
  if cloud_type == "azure":
    return get_creds_from_env_azure()
  elif cloud_type == "huggingface":
    return get_creds_from_env_huggingface()
  elif cloud_type == "sftp":
    return get_creds_from_env_sftp()
  # Add more cloud types here as needed:
  # elif cloud_type == <SOMETHING ELSE>:
  #   return get_creds_from_something_else():
  else:
    raise ValueError(f"Unsupported cloud type: {cloud_type}")


def _init_service(cloud_type: CloudType, cloud_access: CloudAccess, credentials: CloudCredentials) -> CloudService:
  if cloud_type == "azure":
    return init_service_azure(credentials, cloud_access)
  elif cloud_type == "huggingface":
    return init_service_huggingface(credentials, cloud_access)
  elif cloud_type == "sftp":
    return init_service_sftp(credentials, cloud_access)
  # Add more cloud types here as needed:
  # elif cloud_type == <SOMETHING ELSE>:
  #   return init_service_something_else(cloud_type, credentials)
  else:
    raise ValueError(f"Unsupported cloud type: {cloud_type}")
