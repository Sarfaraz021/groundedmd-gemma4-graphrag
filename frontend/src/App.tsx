import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Toaster } from '@/components/ui/sonner';
import { ThemeProvider } from '@/hooks/useTheme';
import Landing from '@/pages/Landing';
import Chat from '@/pages/Chat';
import Ingest from '@/pages/Ingest';
import NotFound from '@/pages/NotFound';

const App = () => {
  return (
    <ThemeProvider>
      <Toaster />
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/app" element={<Chat />} />
          <Route path="/ingest" element={<Ingest />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
