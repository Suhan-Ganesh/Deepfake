# Google OAuth Setup Instructions

To enable Google Authentication in VeriChain, follow these steps:

## 1. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name: "VeriChain" and click "Create"

## 2. Enable Google OAuth

1. In your project, go to **APIs & Services** → **OAuth consent screen**
2. Choose **External** user type → Click "Create"
3. Fill in the required fields:
   - App name: `VeriChain`
   - User support email: Your email
   - Developer contact email: Your email
4. Click "Save and Continue"
5. Skip the "Scopes" section → Click "Save and Continue"
6. Add test users (your email) → Click "Save and Continue"

## 3. Create OAuth Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click "Create Credentials" → "OAuth client ID"
3. Choose application type: **Web application**
4. Name: `VeriChain Web Client`
5. Add Authorized JavaScript origins:
   - `http://localhost:3000`
   - `http://127.0.0.1:3000`
6. Add Authorized redirect URIs:
   - `http://localhost:3000`
   - `http://127.0.0.1:3000`
7. Click "Create"
8. **Copy your Client ID** (looks like: `xxxxx.apps.googleusercontent.com`)

## 4. Update the Code

1. Open `src/components/AuthScreen.js`
2. Find this line:
   ```javascript
   const GOOGLE_CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com";
   ```
3. Replace `YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com` with your actual Client ID
4. Save the file

## 5. Test the Application

1. Start the development server: `npm start`
2. The splash screen should appear, then the auth screen
3. Click "Sign in with Google" and authenticate
4. Connect your MetaMask wallet
5. Click "Continue to App"

## Troubleshooting

- **"Invalid Client" error**: Double-check your Client ID
- **Redirect URI mismatch**: Ensure your authorized URIs match exactly
- **MetaMask not detected**: Install [MetaMask browser extension](https://metamask.io/)

## Security Notes

- Never commit your Client ID to public repositories
- For production, add your production domain to authorized origins
- Keep your OAuth credentials secure
