"use client";

import { useState, useEffect } from "react";

import { useTheme } from "next-themes";



interface SidebarProps {
  onClose: () => void;
}

export default function Sidebar({ }: SidebarProps) {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    if (resolvedTheme) setMounted(true);
  }, [resolvedTheme]);

  const toggleTheme = () => {
    setTheme(resolvedTheme === "dark" ? "light" : "dark");
  };

  

  
  

  if (!mounted) {
    return null; // Avoid hydration mismatch
  }

  return (
    <div className="fixed inset-y-0 left-0 w-full sm:w-80 h-screen bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden">
      {/* Header Section */}
      {/* Footer (Theme Toggle) */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={toggleTheme}
          className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
        >
          <span>{resolvedTheme === "dark" ? "☀️" : "🌙"}</span>
          <span>{resolvedTheme === "dark" ? "Light Mode" : "Dark Mode"}</span>
        </button>
      </div>
     
      {/* <Image
              src="/assets/images/krishna-2.jpg"
              alt="SAMAY"
              width={320}
              height={1000}
              className="object-cover h-full opacity-80 lg:w-[320px] xl:w-[400px] 2xl:w-[480px]"
            /> */}

<div className="w-full h-full flex flex-col items-center justify-center p-6 bg-gradient-to-br from-indigo-500/10 via-purple-500/10 to-pink-500/10 dark:from-indigo-900/30 dark:via-purple-900/30 dark:to-pink-900/30">
  <div className="text-center space-y-6">
    <h3 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 dark:from-indigo-400 dark:via-purple-400 dark:to-pink-400 bg-clip-text text-transparent">
      Wisdom of the Ages
    </h3>
    <div className="space-y-4">
      <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
        Shrimad Bhagavad Gita
      </p>
      <p className="text-sm text-gray-600 dark:text-gray-400 italic">
        The Ultimate Guide to Enlightenment
      </p>
      <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
        Yoga Sutras of Patanjali
      </p>
    </div>
  </div>
</div>
      
    </div>
  );
}
