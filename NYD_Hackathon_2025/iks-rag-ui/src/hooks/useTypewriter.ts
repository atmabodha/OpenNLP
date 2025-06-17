import { useState, useEffect, useRef } from 'react';

export const useTypewriter = (
  text: string, 
  speed: number = 10, 
  delay: boolean = false
) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const [shouldStart, setShouldStart] = useState(!delay);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const indexRef = useRef(0);

  // Handle delay changes
  useEffect(() => {
    if (!delay) {
      setShouldStart(true);
    }
  }, [delay]);

  useEffect(() => {
    if (!text || !shouldStart) {
      return;
    }

    indexRef.current = 0;
    setDisplayedText('');
    setIsComplete(false);

    const timer = setInterval(() => {
      if (indexRef.current <= text.length) {
        setDisplayedText(text.slice(0, indexRef.current));
        indexRef.current += 1;
      } else {
        clearInterval(timer);
        setIsComplete(true);
      }
    }, speed);

    timerRef.current = timer;

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [text, speed, shouldStart]);

  return { displayedText, isComplete };
}; 