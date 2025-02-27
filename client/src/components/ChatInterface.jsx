import React, { useState } from "react";
import CircularProgress from "@mui/material/CircularProgress";
import MessageDisplay from "./MessageDisplay";
import InputSection from "./InputSection";
import ModelDropdown from "./dropdown";
import { useData } from "../context/DataContext";
import axios from "axios";

const ChatInterface = () => {
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const { data, addMessage } = useData();

    const activeChat = data.chatHistory.find(
        (chat) => chat.id === data.activeChatId
    );

    const handleSend = async () => {
        if (!input.trim()) return;

        setLoading(true);

        try {
            const baselineApiUrl = "http://127.0.0.1:5000/baseline";
            const hybridApiUrl = "http://127.0.0.1:5000/hybrid";

            const [baselineResponse, hybridResponse] = await Promise.all([
                axios.post(baselineApiUrl, {
                    text: input,
                    max_length: 512,
                    num_beams: 5,
                    early_stopping: true,
                }),
                axios.post(hybridApiUrl, {
                    text: input,
                    max_length: 512,
                    num_beams: 5,
                    early_stopping: true,
                }),
            ]);

            const combinedResponse = {
                response: {
                    baseline: baselineResponse.data,
                    hybrid: hybridResponse.data,
                },
            };

            const selectedModel = data.selectedModel || "Baseline";
            const generatedText =
                selectedModel === "Hybrid"
                    ? hybridResponse.data.generated_text
                    : baselineResponse.data.generated_text;

            const userMessage = {
                text: input,
                sender: "user",
                id: Date.now(),
                timestamp: new Date().toLocaleString(), // Add timestamp
                botResponse: {
                    text: generatedText,
                    sender: "bot",
                    id: Date.now() + 1,
                    timestamp: new Date().toLocaleString(), // Add timestamp
                    predictions: combinedResponse.response,
                    modelUsed: selectedModel,
                    weightedSum:
                        selectedModel === "Hybrid"
                            ? hybridResponse.data.weighted_sum
                            : baselineResponse.data.weighted_sum,
                },
            };

            addMessage(userMessage);
            setInput("");
        } catch (error) {
            console.error("Error:", error);
            alert(`An error occurred: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full h-full max-h-screen">
            {loading && (
                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm z-50">
                    <CircularProgress size={150} />
                </div>
            )}
            <div className="h-full px-36">
                <div className="flex flex-col h-full gap-5">
                    <div className="flex justify-center">
                        <ModelDropdown />
                    </div>
                    <div className="flex-1 overflow-y-auto scrollbar-hide">
                        <MessageDisplay />
                    </div>
                    <InputSection
                        input={input}
                        setInput={setInput}
                        handleSend={handleSend}
                    />
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;