import { PROMPT_EXPAND_TEST } from "./constants.mts"
import {
    checkLLMResponse,
    modelOptions,
    parseRules,
    parseRulesTests,
} from "./parsers.mts"
import { measure } from "./perf.mts"
import type {
    PromptPexContext,
    PromptPexOptions,
    PromptPexTestEval,
    PromptPexTest,
} from "./types.mts"

const { generator } = env

export async function expandTests(
    files: PromptPexContext,
    options?: PromptPexOptions
) {
    const ruleTests = parseRulesTests(files.tests.content)
    const rules = parseRules(files.rules.content)

    for (let i = 0; i < ruleTests.length; i++) {
        const test = ruleTests[i]
        const res = await measure("expand.test", () =>
            generator.runPrompt(
                (ctx) => {
                    ctx.importTemplate(PROMPT_EXPAND_TEST, {
                        intent: files.intent.content,
                        rules: files.rules.content,
                        test: test.testinput,
                    })
                },
                {
                    ...modelOptions("rules", options),
                    cache: "promptpex",
                    label: `expanding test case ${i + 1}/${ruleTests.length}`,
                }
            )
        )

        if (!res.error) test.testinputexpanded = res.text
    }
}
