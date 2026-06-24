import type {ReactNode} from 'react';
import {NavLink, Outlet} from 'react-router-dom';
import {useAuth} from '../context/useAuth';

function NavIcon({
  to,
  label,
  children,
}: {
  to: string;
  label: string;
  children: ReactNode;
}) {
  return (
    <NavLink
      to={to}
      title={label}
      className={({isActive}) =>
        `flex h-11 w-11 items-center justify-center rounded-xl transition-colors ${
          isActive
            ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
            : 'text-slate-400 hover:bg-slate-100 hover:text-slate-600'
        }`
      }
    >
      {children}
    </NavLink>
  );
}

export function AppShell() {
  const {user, logout} = useAuth();

  return (
    <div className="flex h-screen bg-slate-50">
      <nav className="flex w-[72px] flex-col items-center justify-between border-r border-slate-200 bg-white py-4">
        <div className="flex flex-col items-center gap-3">
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-sm font-semibold text-white">
            C
          </div>
          <NavIcon to="/chat" label="Chats">
            <ChatIcon />
          </NavIcon>
          <NavIcon to="/directories" label="Directories">
            <FolderIcon />
          </NavIcon>
          <NavIcon to="/chunks" label="View chunks">
            <ChunksIcon />
          </NavIcon>
          {user?.role === 'admin' && (
            <NavIcon to="/admin/support" label="Admin support">
              <SupportIcon />
            </NavIcon>
          )}
        </div>
        <div className="flex flex-col items-center gap-3">
          <NavLink
            to="/dashboard"
            className={({isActive}) =>
              `flex h-9 w-9 items-center justify-center rounded-full text-xs font-semibold transition-colors ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`
            }
            title={user?.email}
          >
            {(user?.fullName ?? user?.email ?? '?').slice(0, 1).toUpperCase()}
          </NavLink>
          <button
            onClick={() => void logout()}
            title="Log out"
            className="flex h-9 w-9 items-center justify-center rounded-xl text-slate-400 transition-colors hover:bg-rose-50 hover:text-rose-500"
          >
            <LogoutIcon />
          </button>
        </div>
      </nav>
      <div className="flex flex-1 overflow-hidden">
        <Outlet />
      </div>
    </div>
  );
}

function ChatIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path
        d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path
        d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ChunksIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M4 5h16M4 12h16M4 19h16" strokeLinecap="round" />
      <path d="M8 3v4M16 10v4M11 17v4" strokeLinecap="round" />
    </svg>
  );
}

function SupportIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path
        d="M12 3l7 3v5c0 4.4-2.8 8.3-7 10-4.2-1.7-7-5.6-7-10V6l7-3z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M9 12h6M12 9v6" strokeLinecap="round" />
    </svg>
  );
}

function LogoutIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path
        d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
