import { DOCS_GLOSSARY } from "./constants.mts"
import type { PromptPexOptions } from "./types.mts"
const { output } = env

export function outputGlossary(options: PromptPexOptions) {
    if (!options?.workflowDiagram) return
    output.details(`Glossary`, DOCS_GLOSSARY)
}

export function outputWorkflowDiagram(
    diagram: string,
    options: PromptPexOptions
) {
    if (!options?.workflowDiagram) return

    output.detailsFenced(
        `workflow`,
        `
graph TD
    ${diagram.trim().split(`\n`).join("\n    ")}
`,
        "mermaid"
    )
}

export async function outputPrompty(
    filename: string,
    options: PromptPexOptions
) {
    if (options?.outputPrompts)
        output.detailsFenced(
            filename,
            (await workspace.readText(filename)).content,
            "md"
        )
}

export function outputFile(file: WorkspaceFile) {
    const { output } = env
    const contentType = path.extname(file.filename)
    output.fence(file.content, contentType)
}

export function outputLines(file: WorkspaceFile, name: string, mode: 'rule' | 'inverseRule' = 'rule') {
    const { output } = env
    const { content, filename } = file
    const contentType = path.extname(filename)
  
    let lines: Record<string, string>[] = []
  
    try {
      const parsed = JSON.parse(content)
      if (Array.isArray(parsed)) {
        lines = parsed.map((entry) => ({ [name]: entry[mode] }))
      } else {
        throw new Error("Parsed content is not an array.")
      }
    } catch (e) {
      lines = content?.split("\n").map((line) => ({ [name]: line })) || []
    }
  
    output.table(lines)
    output.detailsFenced(`data`, content, contentType)
  }