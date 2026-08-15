import { authConfigured } from "@/auth";

import LoginClient from "./client";

export default function Login() {
  // Whether Google is wired up is a server-side fact (it depends on env vars),
  // so it is resolved here and handed down rather than probed in the browser.
  return <LoginClient googleConfigured={authConfigured} />;
}
