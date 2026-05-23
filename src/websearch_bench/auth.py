"""Shared credential helper.

Uses :class:`DefaultAzureCredential` but excludes managed identity / VS Code /
shared-token-cache sources that aren't expected on a local dev laptop. This
eliminates the IMDS metadata probe (which always 504s off-Azure and shows up
as a red ``GET /metadata/identity/oauth2/token`` dependency in App Insights).

Order tried (top to bottom):
  1. Environment variables (AZURE_CLIENT_ID/SECRET/TENANT_ID)
  2. Azure CLI (``az login``)
  3. Azure Developer CLI (``azd auth login``)
  4. Azure PowerShell
"""

from __future__ import annotations

from azure.identity.aio import DefaultAzureCredential


def make_credential() -> DefaultAzureCredential:
    """DefaultAzureCredential with IMDS / VS Code / cache probes disabled."""
    return DefaultAzureCredential(
        exclude_managed_identity_credential=True,
        exclude_visual_studio_code_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_workload_identity_credential=True,
    )
