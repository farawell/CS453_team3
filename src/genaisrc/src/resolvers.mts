import type {
    PromptPexContext,
    PromptPexRule,
    PromptPexTest,
    PromptPexTestGenerationScenario,
} from "./types.mts"
const dbg = host.logger("promptpex:resolvers")

export async function resolveTestId(
    files: PromptPexContext,
    test: PromptPexTest
) {
    const content = MD.content(files.prompt.content)
    const testid = await parsers.hash(
        content + test.testinput + (test.baseline ? ";baseline" : ""),
        {
            length: 7,
        }
    )
    return testid
}

export function resolveScenarios(
    files: PromptPexContext
): PromptPexTestGenerationScenario[] {
    return (
        files.frontmatter?.scenarios || [
            {
                name: "",
                instructions: "",
                parameters: {},
            } satisfies PromptPexTestGenerationScenario,
        ]
    )
}

export function resolvePromptArgs(
    files: PromptPexContext,
    test: PromptPexTest
) {
    const { inputs } = files
    const { testinput, expectedoutput, scenario } = test

    const inputKeys = Object.keys(inputs)
    const unresolved = new Set(inputKeys)
    dbg(`unresolved inputs: %s`, inputKeys)
    const args: Record<string, any> = {}

    // apply scenario values
    if (scenario) {
        const scenarios = resolveScenarios(files)
        const scenarioInstance = scenarios.find((s) => s.name === scenario)
        if (!scenarioInstance) throw new Error(`Scenario ${scenario} not found`)
        for (const [iname, ivalue] of Object.entries(
            scenarioInstance.parameters || {}
        )) {
            if (unresolved.has(iname) && ivalue !== undefined) {
                dbg(`input %s scenario: %s`, iname, ivalue)
                args[iname] = ivalue
                unresolved.delete(iname)
            }
        }
    }

    // apply defaults
    for (const [iname, ivalue] of Object.entries(inputs)) {
        if (unresolved.has(iname) && ivalue?.default !== undefined) {
            dbg(`input %s default: %s`, iname, ivalue.default)
            args[iname] = ivalue.default
            unresolved.delete(iname)
        }
    }

    // fill last whole with generated input
    dbg(`remaining unresolved inputs: %s`, Array.from(unresolved))
    if (unresolved.size === 1) {
        const key = Array.from(unresolved)[0]
        dbg(`input %s <- %s`, key, testinput)
        args[key] = testinput
    } else if (unresolved.size > 1) {
        const parsedInputs = parsers.JSON5(testinput)
        dbg(`parsed inputs: %O`, parsedInputs)
        if (typeof parsedInputs !== "object") {
            dbg(`invalid test input format: %o`, testinput)
            throw new Error(
                `Invalid test input format, expected JSON object: ${testinput}`
            )
        }
        Object.assign(args, parsedInputs)
    } else if (unresolved.size === 0 && inputKeys.length > 0) {
        dbg(
            `all inputs prefilled, replacing first (%s) with test input`,
            inputKeys[0]
        )
        args[inputKeys[0]] = testinput
    }
    return {
        inputs,
        args,
        testInput: testinput,
        expectedOutput: expectedoutput,
    }
}

export async function resolveRuleHash(files: PromptPexContext, rule: string) {
    const content = MD.content(files.prompt.content)
    const ruleId = await parsers.hash(content + rule, {
        length: 7,
    })
    return ruleId
}

/* FIXME: not anymore doing resolve rule */
export function resolveRule(
    rules: PromptPexRule[]
): { id: string; rule: string }[] {
    const output: { id: string; rule: string }[] = []

    for (const r of rules) {
        output.push({
            id: r.id,
            rule: r.inversed ? r.inverseRule : r.rule,
        })
    }
    return output
}

export async function resolvePromptId(files: PromptPexContext) {
    const content = MD.content(files.prompt.content)
    return parsers.hash(content, { length: 7 })
}
