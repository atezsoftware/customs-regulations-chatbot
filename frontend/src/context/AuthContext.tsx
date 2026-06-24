import {useCallback, useEffect, useState} from 'react';
import type {ReactNode} from 'react';
import {authApi} from '../lib/endpoints';
import {tokenStore} from '../lib/tokenStore';
import type {SafeUser} from '../types';
import {AuthContext} from './auth-context';

export function AuthProvider({children}: {children: ReactNode}) {
  const [user, setUser] = useState<SafeUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (!tokenStore.getAccess()) {
        setLoading(false);
        return;
      }
      authApi
        .me()
        .then(setUser)
        .catch(() => tokenStore.clear())
        .finally(() => setLoading(false));
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const {user, tokens} = await authApi.login({email, password});
    tokenStore.set(tokens);
    setUser(user);
  }, []);

  const register = useCallback(
    async (email: string, password: string, fullName?: string) => {
      const {user, tokens} = await authApi.register({email, password, fullName});
      tokenStore.set(tokens);
      setUser(user);
    },
    [],
  );

  const logout = useCallback(async () => {
    const refresh = tokenStore.getRefresh();
    tokenStore.clear();
    setUser(null);
    if (refresh) {
      await authApi.logout(refresh).catch(() => {
        // best-effort revoke; local session is already cleared either way
      });
    }
  }, []);

  return (
    <AuthContext.Provider value={{user, loading, login, register, logout}}>
      {children}
    </AuthContext.Provider>
  );
}
