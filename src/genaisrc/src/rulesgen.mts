import {
    RULES_NUM,
    PROMPT_GENERATE_OUTPUT_RULES,
    DIAGRAM_GENERATE_OUTPUT_RULES,
} from "./constants.mts"
import { outputWorkflowDiagram, outputPrompty } from "./output.mts"
import {
    modelOptions,
    checkLLMResponse,
    tidyRules,
    parseRules,
} from "./parsers.mts"
import { measure } from "./perf.mts"
import type {
    PromptPexContext,
    PromptPexOptions,
    PromptPexRule,
} from "./types.mts"
const dbg = host.logger("promptpex:gen:rules")
const { generator } = env

export async function generateOutputRules(
    files: PromptPexContext,
    options?: PromptPexOptions & { numRules?: number }
): Promise<void> {
    const { numRules = RULES_NUM, rulesModel = "rules" } = options || {}

    dbg(`generating ${numRules} output rules`)
    const instructions =
        options?.instructions?.outputRules ||
        files.frontmatter?.instructions?.outputRules ||
        ""

    outputWorkflowDiagram(DIAGRAM_GENERATE_OUTPUT_RULES, options)

    // generate rules
    const input_data = MD.content(files.prompt.content)
    const pn = PROMPT_GENERATE_OUTPUT_RULES
    await outputPrompty(pn, options)
    const res = await measure("gen.outputrules", () =>
        generator.runPrompt(
            (ctx) => {
                ctx.importTemplate(pn, {
                    num_rules: numRules,
                    input_data,
                    instructions,
                })
            },
            {
                ...modelOptions(rulesModel, options),
                //      logprobs: true,
                label: `${files.name}> generate rules`,
            }
        )
    )
    const rulesText = tidyRules(checkLLMResponse(res))
    const rulesArray = parseRules(rulesText)
    const pairedRules: PromptPexRule[] = rulesArray.map((r, i) => ({
        id: `rule_${i + 1}`,
        rule: r,
        inverseRule: "", // Placeholder for inverse rule, filled in next step
        inversed: false,
    }))
    files.rules.content = JSON.stringify(pairedRules, null, 2)
    if (files.writeResults) await workspace.writeFiles([files.rules])
}
