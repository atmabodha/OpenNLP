"use client";

import { useTheme } from "next-themes";

interface NavbarProps {
  onToggleSidebar: () => void;
}



export default function Navbar({ onToggleSidebar }: NavbarProps) {
  const { theme, setTheme } = useTheme();
  
  const toggleTheme = () => {
    if (theme === 'dark') {
      setTheme('light');
    } else {
      setTheme('dark');
    }
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50">
      {/* Gradient line at the top */}
      <div className="h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500" />
      
      {/* Main navbar content */}
      <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200/80 dark:border-gray-800/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Left section */}
            <div className="flex items-center space-x-4">
              <button
                onClick={onToggleSidebar}
                className="p-2.5 rounded-xl text-gray-500 dark:text-gray-400
                         hover:bg-gray-100 dark:hover:bg-gray-800
                         transition-all duration-200 hover:scale-105"
                aria-label="Toggle sidebar"
              >
                <span className="text-xl">📚</span>
              </button>
              <div className="flex items-center space-x-3">
                <div className="flex items-center">
                  <span className="text-2xl font-bold bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 
                               text-transparent bg-clip-text hover:from-pink-500 hover:via-purple-500 hover:to-indigo-500 
                               transition-all duration-500 cursor-default">
                    SAMAY
                  </span>
                  <div className="relative">
                    <span className="absolute -right-9 -top-3 hidden sm:inline-block px-2 py-1 text-xs font-medium 
                                 text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/30 
                                 rounded-full transform -rotate-12 hover:rotate-0 transition-transform duration-200">
                      Beta
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right section */}
            <div className="flex items-center space-x-6">
              <div className="hidden sm:flex items-center space-x-1">
                <a 
                  href="#" 
                  className="px-4 py-2 rounded-xl text-sm font-medium text-gray-600 dark:text-gray-300
                            hover:bg-gray-100 dark:hover:bg-gray-800 transition-all duration-200
                            hover:scale-105 hover:text-indigo-600 dark:hover:text-indigo-400"
                >
                  <span className="flex items-center space-x-2">
                    <span>🔍</span>
                    <span>About</span>
                  </span>
                </a>
                <a 
                  href="#" 
                  className="px-4 py-2 rounded-xl text-sm font-medium text-gray-600 dark:text-gray-300
                            hover:bg-gray-100 dark:hover:bg-gray-800 transition-all duration-200
                            hover:scale-105 hover:text-indigo-600 dark:hover:text-indigo-400"
                >
                  <span className="flex items-center space-x-2">
                    <span>💡</span>
                    <span>Help</span>
                  </span>
                </a>
              </div>

              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className="p-2.5 rounded-xl bg-gray-100 dark:bg-gray-800 
                         hover:bg-gray-200 dark:hover:bg-gray-700
                         transition-all duration-200 hover:scale-105
                         shadow-sm hover:shadow-md"
                aria-label="Toggle theme"
              >
                <span className="text-xl">
                  {theme === 'dark' ? '🌞' : '🌙'}
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
} 


