import {
    TESTS_NUM,
    PROMPT_GENERATE_TESTS,
    DIAGRAM_GENERATE_TESTS,
    GENERATION_SYMBOL,
    SCENARIO_SYMBOL,
    RULE_SYMBOL,
} from "./constants.mts"
import { outputWorkflowDiagram, outputPrompty } from "./output.mts"
import {
    parseAllRules,
    modelOptions,
    isUnassistedResponse,
} from "./parsers.mts"
import { measure } from "./perf.mts"
import { resolveScenarios } from "./resolvers.mts"
import { convertToTestData } from "./testdata.mts"
import type {
    PromptPexContext,
    PromptPexOptions,
    PromptPexRule,
    PromptPexTest,
} from "./types.mts"
const dbg = host.logger("promptpex:gen:test")
const { generator, output } = env

export async function generateTests(
    files: PromptPexContext,
    options?: PromptPexOptions
): Promise<PromptPexTest[]> {
    const {
        testsPerRule: num = TESTS_NUM,
        rulesModel = "rules",
        testGenerations = 1,
    } = options || {}

    if (!files.rules.content) throw new Error("No rules found")
    if (!files.inputSpec.content) throw new Error("No input spec found")
    const allRules = parseAllRules(files, options)
    dbg(`rules: ${allRules.length}`)
    if (!allRules) throw new Error("No rules found")

    outputWorkflowDiagram(DIAGRAM_GENERATE_TESTS, options)

    const scenarios = resolveScenarios(files)
    dbg(`scenarios: %d`, scenarios.length)
    const context = MD.content(files.prompt.content)
    const pn = PROMPT_GENERATE_TESTS
    // TODO: parameterize how many and which test samples to use
    const testSamples = (files.testSamples || []).slice(0, 5)
    const test_samples = testSamples?.length ? YAML.stringify(testSamples) : ""
    dbg(`test samples: %d`, test_samples)

    await outputPrompty(pn, options)

    const mutatedRules = mutateRules(allRules, files, options)
    const tests: PromptPexTest[] = []

    const checkpoint = async () => {
        dbg(`saving ${tests.length} tests`)
        const resc = JSON.stringify(tests, null, 2)
        files.tests.content = resc
        if (files.writeResults) await workspace.writeFiles(files.tests)
        await convertToTestData(files, tests)
    }

    dbg(`rule groups: ${mutatedRules.length}`)
    for (let si = 0; si < scenarios.length; si++) {
        const scenario = scenarios[si]
        const { instructions, parameters = {} } = scenario

        dbg(`scenario: ${scenario.name}`)
        let testGeneration = 0
        let repaired = false
        const rule = mutatedRules
            .map((r, index) => `${index + 1}. ${r.rule}`)
            .join("\n")
        dbg(`rule: ${rule}`)
        const scenarioInstructions = [
            instructions,
            ...Object.entries(parameters).map(([k, v]) => `'{{${k}}}' = ${v}`),
        ]
            .filter((l) => !!l)
            .map((l) => `- ${l}`)
            .join("\n")

        const args = Object.fromEntries(
            Object.entries(files.inputs).filter(([k]) => !parameters[k])
        )
        dbg(`open args: %O`, args)

        const argNames = Object.keys(args)
        const testinput_names = argNames.join(", ")
        const testinput_descriptions = Object.entries(args)
            .map(
                ([k, v]) =>
                    `- "${k}": ${v.type} = ${v.description || `Detailed input '${k}' provided to the software.`}`
            )
            .join("\n")
        const testinput_example_1 = argNames
            .map((name) => `input '${name}' for scenario 1`)
            .join(",")
        const testinput_example_2 = argNames
            .map((name) => `input '${name}' for scenario 2`)
            .join(",")
        const testinput_count = argNames.length

        dbg(`testinput: %O`, {
            testinput_names,
            testinput_descriptions,
            testinput_example_1,
            testinput_example_2,
            testinput_count,
        })

        await measure("gen.tests", () =>
            generator.runPrompt(
                (ctx) => {
                    ctx.importTemplate(pn, {
                        input_spec: files.inputSpec.content,
                        context,
                        num,
                        scenario: scenarioInstructions,
                        rule,
                        num_rules: mutatedRules.length,
                        testinput_descriptions,
                        testinput_names,
                        testinput_example_1,
                        testinput_example_2,
                        testinput_count,
                        test_samples,
                    })
                    ctx.defChatParticipant((p, c) => {
                        const last: string = c.at(-1)?.content as string
                        dbg(`last message: %s`, last)
                        if (typeof last !== "string")
                            throw new Error("Invalid last message")
                        const csv = parseCsvTests(last, Object.keys(args))
                        if (!csv.length) {
                            if (!repaired) {
                                dbg(`no tests found, trying to repair`)
                                console.warn(
                                    "Invalid generated test format or no test generated, trying to repair"
                                )
                                repaired = true
                                p.$`The generated tests are not valid CSV. Please fix formatting issues and try again.`
                            } else {
                                output.warn(
                                    "Invalid generated test format, skipping repair."
                                )
                                output.fence(last, "txt")
                            }
                        } else {
                            if (csv?.length) {
                                dbg(`adding ${csv.length} tests`)
                                tests.push(
                                    ...csv.map((t) => ({
                                        ...t,
                                        scenario: scenario.name,
                                        generation: testGeneration,
                                    }))
                                )
                            }
                            testGeneration++
                            if (testGeneration < testGenerations) {
                                dbg(`next test generation ${testGeneration}`)
                                repaired = false
                                p.$`Generate ${num} more tests for the same rules. Do not duplicate the previous tests.`
                            }
                        }
                    })
                },
                {
                    ...modelOptions(rulesModel, options),
                    label: `${files.name}> generate tests (${SCENARIO_SYMBOL} ${scenario.name} ${si + 1}/${scenarios.length}, ${RULE_SYMBOL} ${GENERATION_SYMBOL} ${testGeneration + 1}/${testGenerations})`,
                }
            )
        )
        await checkpoint()
        // TODO retry
    }

    await checkpoint()
    return tests
}

function mutateRules(
    rules: PromptPexRule[],
    files: PromptPexContext,
    options?: PromptPexOptions
): { id: string; rule: string }[] {
    const { mutateRule } = options || {}
    const output: { id: string; rule: string }[] = []
    /* Find the index of the currently inversed rule, if any */
    const currentInversedIdx = rules.findIndex((r) => r.inversed)

    if (mutateRule) {
        /* Determine the next rule to inverse:
         * If none are currently inversed, start with the first rule (index 0),
         * otherwise move to the next rule in the list.
         */
        const nextIndex = currentInversedIdx === -1 ? 0 : currentInversedIdx + 1

        /* Reset all rules to not inversed */
        rules.forEach((r) => (r.inversed = false))

        /* Set the next rule as inversed if within bounds */
        if (nextIndex < rules.length) {
            rules[nextIndex].inversed = true
        }

        if (files.writeResults) workspace.writeFiles([files.rules])
    }

    /* Generate output list:
     * Include the inverseRule only for the currently inversed rule,
     * all others should return the original rule.
     */
    for (const r of rules) {
        output.push({
            id: r.id,
            rule: r.inversed ? r.inverseRule : r.rule,
        })
    }

    return output
}

function chunkArray<T>(array: T[], n: number): T[][] {
    const result: T[][] = []
    for (let i = 0; i < array.length; i += n) {
        result.push(array.slice(i, i + n))
    }
    return result
}

function parseCsvTests(
    text: string,
    testInputNames: string[]
): PromptPexTest[] {
    if (!text) return []

    const content = parsers.unfence(text.trim().replace(/\\"/g, '""'), "csv")
    const rulesTests = content
        ? (parsers.CSV(content, {
              delimiter: ",",
              repair: true,
          }) as PromptPexTest[])
        : []
    const res = rulesTests
        .map((r) => {
            const testinput: Record<string, string> = {}
            for (const testInputName of testInputNames) {
                const v = r[testInputName]
                if (v === undefined) {
                    dbg(`testinput %s not found`, testInputName)
                    return undefined // skip test if not found?
                }
                testinput[testInputName] = v
            }
            return {
                ...r,
                testinput:
                    testInputNames.length > 1
                        ? JSON.stringify(testinput)
                        : testinput[testInputNames[0]],
            }
        })
        .filter((t) => !!t)
    if (!res.length)
        output.detailsFenced(`tests - unable to parse`, text, "text")
    return res
}
