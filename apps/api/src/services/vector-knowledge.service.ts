export interface KnowledgeChunk {
  courseSlug: string;
  content: string;
  source: string;
  score?: number;
}

export class VectorKnowledgeService {
  async retrieveContext(courseSlug: string, query: string): Promise<KnowledgeChunk[]> {
    return [
      {
        courseSlug,
        content: `Vector-search placeholder for '${query}'. Connect LangChain retrievers and a managed vector database here for production course knowledge grounding.`,
        source: "course-knowledge-base"
      }
    ];
  }
}
