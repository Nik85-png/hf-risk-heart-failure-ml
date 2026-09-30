"""
authenticate_adc.py
Authenticates with Google Cloud to create Application Default Credentials (ADC)
for PhysioNet BigQuery MIMIC-IV queries.

SECURITY NOTE
-------------
The OAuth client id and secret are read from the GOOGLE_OAUTH_CLIENT_ID and
GOOGLE_OAUTH_CLIENT_SECRET environment variables. They are deliberately NOT
stored in this file: a credential committed to git history is a live
exposure and must be revoked/rotated if it was ever committed, not merely
deleted from the working tree.
"""

import json
import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

def build_client_config():
    """
    Build the OAuth client config from environment variables.

    SECURITY: the OAuth client id/secret are deliberately NOT stored in this
    file. A credential committed to git is a live exposure; if one was ever
    committed it must be revoked/rotated, not just deleted from the working
    tree.
    """
    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit(
            "Missing OAuth client credentials.\n"
            "Set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET in "
            "your environment first (values from your Google Cloud console "
            "OAuth client).\n"
            "Example (Git Bash):\n"
            "  export GOOGLE_OAUTH_CLIENT_ID=....apps.googleusercontent.com\n"
            "  export GOOGLE_OAUTH_CLIENT_SECRET=..."
        )
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url":
                "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": ["http://localhost"],
        }
    }

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]

PROJECT_ID = "gen-lang-client-0398113511"


def main():
    print("=" * 60)
    print(" MIMIC-IV BigQuery Authentication (ADC Setup)")
    print("=" * 60)
    print("\nA browser window will open automatically.")
    print("Please select your Google account (nikpro0805@gmail.com)")
    print("with MIMIC-IV / PhysioNet access and click 'Allow'.\n")

    try:
        client_config = build_client_config()
        flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
        creds = flow.run_local_server(
            port=0,
            prompt="consent",
            authorization_prompt_message="Waiting for browser authentication...\nURL: {url}",
            success_message="Authentication successful! You may close this browser tab.",
            open_browser=True,
        )

        appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
        adc_dir = os.path.join(appdata, "gcloud")
        os.makedirs(adc_dir, exist_ok=True)
        adc_path = os.path.join(adc_dir, "application_default_credentials.json")

        adc_data = {
            "account": "",
            "client_id": client_config["installed"]["client_id"],
            "client_secret": client_config["installed"]["client_secret"],
            "quota_project_id": PROJECT_ID,
            "refresh_token": creds.refresh_token,
            "type": "authorized_user",
        }

        with open(adc_path, "w", encoding="utf-8") as f:
            json.dump(adc_data, f, indent=2)

        print("\n" + "=" * 60)
        print("✓ SUCCESS: Application Default Credentials written to:")
        print(f"  {adc_path}")
        print(f"✓ Quota project set to: {PROJECT_ID}")
        print("=" * 60)
        print("\nYou can now run the extraction script!")
        return 0

    except Exception as e:
        print(f"\n❌ Error during authentication: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
