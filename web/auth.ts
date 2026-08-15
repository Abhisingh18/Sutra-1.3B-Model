import NextAuth from "next-auth";
import Google from "next-auth/providers/google";

/* Google sign-in, configured only when the credentials exist.
 *
 * The provider list is built conditionally on purpose. A deployment without
 * AUTH_GOOGLE_ID would otherwise fail at build time, taking the whole site
 * down over an optional feature -- and sign-in is optional here: the chat
 * works anonymously, and an account only separates saved conversations.
 */

export const authConfigured = Boolean(
  process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET
);

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: authConfigured
    ? [
        Google({
          clientId: process.env.AUTH_GOOGLE_ID,
          clientSecret: process.env.AUTH_GOOGLE_SECRET,
        }),
      ]
    : [],
  pages: { signIn: "/login" },
  session: { strategy: "jwt" },
  callbacks: {
    // No database. The JWT carries the profile in a cookie, which is all the
    // UI needs: a name, a picture, and a stable id to key saved chats against.
    async session({ session, token }) {
      if (session.user && token.sub) session.user.id = token.sub;
      return session;
    },
  },
  trustHost: true,
});
