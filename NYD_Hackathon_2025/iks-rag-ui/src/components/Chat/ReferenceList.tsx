import { Reference } from '@/types/chat';

interface ReferenceListProps {
  references: Reference[];
  isUserMessage: boolean;
}

export default function ReferenceList({ references }: ReferenceListProps) {
  return (
    <div className="mt-2 text-sm">
      <p className="font-semibold text-muted-foreground">References:</p>
      <ul className="list-disc list-inside space-y-1">
        {references.map((ref, index) => (
          <li key={index} className="text-muted-foreground">
            <a 
              href={ref.link}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline transition-colors duration-200"
              aria-label={`Reference from ${ref.source}`}
            >
              {ref.source}
              {ref.text && <span className="text-muted-foreground"> - {ref.text}</span>}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
