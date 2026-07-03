import {Navigate, Outlet} from 'react-router-dom';
import {useAuth} from '../context/useAuth';
import {Spinner} from './ui/Spinner';

export function LocalOnlyRoute() {
  const {user, loading} = useAuth();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center text-slate-400">
        <Spinner />
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;
  if (!user.uploadsEnabled) return <Navigate to="/chat" replace />;

  return <Outlet />;
}
