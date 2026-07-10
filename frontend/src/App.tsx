import {Navigate, Route, Routes} from 'react-router-dom';
import {AdminRoute} from './components/AdminRoute';
import {AppShell} from './components/AppShell';
import {LocalOnlyRoute} from './components/LocalOnlyRoute';
import {ProtectedRoute} from './components/ProtectedRoute';
import {AdminAmendmentsPage} from './pages/AdminAmendmentsPage';
import {AdminSupportPage} from './pages/AdminSupportPage';
import {ChatPage} from './pages/ChatPage';
import {ChunksPage} from './pages/ChunksPage';
import {DashboardPage} from './pages/DashboardPage';
import {DirectoriesPage} from './pages/DirectoriesPage';
import {LoginPage} from './pages/LoginPage';
import {RegisterPage} from './pages/RegisterPage';

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      <Route element={<ProtectedRoute />}>
        <Route element={<AppShell />}>
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/chat/:sessionId" element={<ChatPage />} />
          <Route element={<LocalOnlyRoute />}>
            <Route path="/directories" element={<DirectoriesPage />} />
          </Route>
          <Route path="/chunks" element={<ChunksPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route element={<AdminRoute />}>
            <Route path="/admin/support" element={<AdminSupportPage />} />
            <Route path="/admin/amendments" element={<AdminAmendmentsPage />} />
          </Route>
        </Route>
      </Route>

      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
}

export default App;
