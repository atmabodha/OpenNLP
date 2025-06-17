'use client';

import { useState } from 'react';
import Navbar from "@/components/Navbar/Navbar";

export default function LayoutClient({ children }: { children: React.ReactNode }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const handleToggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <>
      <Navbar onToggleSidebar={handleToggleSidebar} />
      <div className="mt-8 py-8 mb-16">
        {children}
      </div>
    </>
  );
} 