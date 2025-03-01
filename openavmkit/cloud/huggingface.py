import os
from datetime import datetime, timezone
import requests
from huggingface_hub import hf_hub_url, upload_file as hf_upload_file
from huggingface_hub.hf_api import HfApi, RepoFolder
from openavmkit.cloud.base import CloudCredentials, CloudService, CloudFile, CloudAccess


class HuggingFaceCredentials(CloudCredentials):

  def __init__(self, token: str):
    super().__init__()
    self.token = token


class HuggingFaceService(CloudService):

  def __init__(
      self,
      credentials: HuggingFaceCredentials,
      repo_id: str,
      access: CloudAccess,
      revision: str = "main"
  ):
    super().__init__("huggingface", credentials, access)
    self.repo_id = repo_id
    self.revision = revision
    self.token = credentials.token
    self.api = HfApi()


  def list_files(self, remote_path: str) -> list[CloudFile]:
    infos = self.api.list_repo_tree(
      repo_id=self.repo_id,
      revision=self.revision,
      token=self.token,
      path_in_repo=remote_path,
      repo_type="dataset",
      recursive=True,
      expand=True
    )

    files = []
    for info in infos:

      if isinstance(info, RepoFolder):
        continue

      if info.rfilename.startswith(remote_path):
        last_modified_date: datetime = info.last_commit.date
        last_modified_utc = last_modified_date.astimezone(timezone.utc)
        files.append(CloudFile(
          name=info.rfilename,
          last_modified_utc=last_modified_utc,
          size=info.size
        ))
    return files


  def download_file(self, remote_file: CloudFile, local_file_path: str):
    super().download_file(remote_file, local_file_path)
    url = hf_hub_url(
      repo_id=self.repo_id,
      filename=remote_file.name,
      repo_type="dataset",
      revision=self.revision
    )
    headers = {"authorization": f"Bearer {self.token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(local_file_path, "wb") as f:
      f.write(response.content)


  def upload_file(self, remote_file_path: str, local_file_path: str):
    super().upload_file(remote_file_path, local_file_path)
    hf_upload_file(
      path_or_fileobj=local_file_path,
      path_in_repo=remote_file_path,
      repo_id=self.repo_id,
      token=self.token,
      repo_type="dataset",
      commit_message="Upload via OpenAVMKit"
    )


def init_service_huggingface(credentials: HuggingFaceCredentials, access: CloudAccess):
  repo_id = os.getenv("HF_REPO_ID")
  if repo_id is None:
    raise ValueError("Missing 'HF_REPO_ID' in environment")
  revision = os.getenv("HF_REVISION", "main")
  service = HuggingFaceService(credentials, repo_id, access, revision)
  return service


def get_creds_from_env_huggingface() -> HuggingFaceCredentials:
  token = os.getenv("HF_TOKEN")
  if not token:
    raise ValueError("Missing 'HF_TOKEN' in environment")
  return HuggingFaceCredentials(token)