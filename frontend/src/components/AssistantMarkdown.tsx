import DOMPurify from 'dompurify';
import type { Components } from 'react-markdown';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

const markdownComponents: Components = {
  a: ({ node: _node, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer" className="text-primary underline underline-offset-2" />
  ),
  mark: ({ node: _node, ...props }) => (
    <mark {...props} className="bg-yellow-200 dark:bg-yellow-400/40 text-inherit rounded px-0.5" />
  ),
};

const prose =
  'prose prose-sm dark:prose-invert max-w-none text-foreground ' +
  'prose-p:my-2 prose-headings:mt-4 prose-headings:mb-2 prose-headings:font-semibold ' +
  'prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-strong:text-foreground ' +
  'prose-blockquote:border-primary/40 prose-blockquote:text-muted-foreground';

// Convert ==text== to <mark>text</mark>, then sanitize — only <mark> tags are allowed through
function applyHighlights(content: string | null | undefined): string {
  if (!content) return '';
  const withMarks = content.replace(/==(.+?)==/gs, '<mark>$1</mark>');
  return DOMPurify.sanitize(withMarks, { ALLOWED_TAGS: ['mark'], ALLOWED_ATTR: [] });
}

interface Props {
  content: string;
  className?: string;
}

export function AssistantMarkdown({ content, className }: Props) {
  return (
    <div className={`${prose} ${className ?? ''}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={markdownComponents}
      >
        {applyHighlights(content)}
      </ReactMarkdown>
    </div>
  );
}
