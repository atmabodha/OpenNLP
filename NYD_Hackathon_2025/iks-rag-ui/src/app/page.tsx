"use client";

import { useEffect, useState, useCallback } from "react";
import Chat from "@/components/Chat/Chat";
import Sidebar from "@/components/Sidebar/Sidebar";

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsSidebarOpen(false);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleCloseSidebar = useCallback(() => setIsSidebarOpen(false), []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-gray-100 to-gray-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 h-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center px-4 z-30">
        <button
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <h1 className="ml-4 text-xl font-semibold">SAMAY</h1>
      </div>

      <div className="flex h-screen pt-16 lg:pt-0">
        {/* Sidebar */}
        <div
          className={`fixed lg:relative z-40 h-[calc(100vh-4rem)] lg:h-screen w-[320px] transform transition-transform duration-200 ease-in-out
            ${isSidebarOpen ? "translate-x-0" : "-translate-x-full"} lg:translate-x-0`}
        >
          <Sidebar onClose={handleCloseSidebar} />
        </div>

        {/* Overlay */}
        {isSidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-30 lg:hidden transition-opacity"
            onClick={handleCloseSidebar}
          />
        )}

        {/* Main Content */}
        <main className="flex-1 relative overflow-hidden">
          <Chat />
        </main>
      </div>
    </div>
  );
}
