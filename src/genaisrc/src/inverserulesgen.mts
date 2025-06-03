import { PROMPT_GENERATE_INVERSE_RULES } from "./constants.mts"
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
const { generator } = env

export async function generateInverseOutputRules(
    files: PromptPexContext,
    options?: PromptPexOptions
): Promise<void> {
    const { rulesModel = "rules" } = options || {}
    const instructions =
        options?.instructions?.inverseOutputRules ||
        files.frontmatter?.instructions?.inverseOutputRules ||
        ""
    outputWorkflowDiagram(
        `OR["Output Rules (OR)"]
IOR["Inverse Output Rules (IOR)"]
OR --> IOR    
`,
        options
    )

    const pairedRules: PromptPexRule[] = JSON.parse(files.rules.content)
    const outputRules = pairedRules.map((p) => p.rule)
    const ruleMarkdown = outputRules.map((r, i) => `${i + 1}. ${r}`).join("\n")

    const pn = PROMPT_GENERATE_INVERSE_RULES
    await outputPrompty(pn, options)

    let inverseRulesText: string
    let inverseRules: string[]
    let retry = 3
    do {
        const res = await measure("gen.inverseoutputrules", () =>
            generator.runPrompt(
                (ctx) => {
                    ctx.importTemplate(pn, {
                        rule: ruleMarkdown,
                        instructions,
                    })
                },
                {
                    ...modelOptions(rulesModel, options),
                    //      logprobs: true,
                    label: `${files.name}> inverse rules`,
                }
            )
        )
        inverseRulesText = tidyRules(checkLLMResponse(res))
        inverseRules = parseRules(inverseRulesText)
    } while (retry-- > 0 && inverseRules.length !== outputRules.length)

    if (inverseRules.length !== outputRules.length) {
        console.warn(
            `inverse rules length mismatch: generated ${inverseRules.length}, expected ${outputRules.length}`
        )
    }
    for (let i = 0; i < pairedRules.length; i++) {
        pairedRules[i].inverseRule = inverseRules[i] || ""
    }
    files.rules.content = JSON.stringify(pairedRules, null, 2)
    if (files.writeResults) await workspace.writeFiles([files.rules])
}
