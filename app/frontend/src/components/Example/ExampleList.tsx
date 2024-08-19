import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "What is linear programming?",
    "How to calculate breakeven point?",
    "What are the main steps involved in preparing a zero-based budget?"
];

const GPT4V_EXAMPLES: string[] = [
    "What is linear programming?",
    "How to calculate breakeven point?",
    "What are the main steps involved in preparing a zero-based budget?"
];

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked, useGPT4V }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {(useGPT4V ? GPT4V_EXAMPLES : DEFAULT_EXAMPLES).map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
