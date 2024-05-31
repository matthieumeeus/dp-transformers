import json

from dataclasses import dataclass
from typing import Dict
from subprocess import check_output
from typing import Optional
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from functools import cached_property


def get_cli_workspace_details() -> Dict[str, str]:
    subscription_id = json.loads(check_output(["az", "account", "show", "--query", "id"]))

    output = json.loads(check_output(["az", "configure", "--list-defaults"]))
    resource_group = next(item for item in output if item["name"] == "group")["value"]
    workspace_name = next(item for item in output if item["name"] == "workspace")["value"]

    return {"subscription_id": subscription_id, "resource_group": resource_group, "workspace_name": workspace_name}


def make_name_aml_safe(name: str) -> str:
    return name.replace("/", "-").replace(".", "_")


@dataclass
class AmlArguments:
    workspace: Optional[str] = None
    resource_group: Optional[str] = None
    subscription_id: Optional[str] = None

    def __post_init__(self):
        cli_details = get_cli_workspace_details()
        self.workspace = self.workspace or cli_details["workspace_name"]
        self.resource_group = self.resource_group or cli_details["resource_group"]
        self.subscription_id = self.subscription_id or cli_details["subscription_id"]

        if not self.workspace or not self.resource_group or not self.subscription_id:
            raise ValueError("Could not find workspace details. Please provide them as arguments or run `az configure`")

    @cached_property
    def ml_client(self) -> MLClient:
        credential = DefaultAzureCredential()
        ml_client = MLClient(credential=credential, workspace_name=self.workspace, resource_group_name=self.resource_group,
                            subscription_id=self.subscription_id)
        return ml_client
