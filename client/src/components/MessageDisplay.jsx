import { useEffect } from "react";
import UserMessage from "./UserMessage";
import BotMessage from "./BotMessage";
import PropTypes from "prop-types";
import { useData } from "../context/DataContext"; // Import useData

const MessageDisplay = () => {
    const { data, setData } = useData(); // Get data and setData from context
    const messages = data.messages || []; // Access messages from context

    const handleSelectOption = (messageId, selectedResponseId) => {
        const updatedMessages = messages.map((message) => {
            if (message.id === messageId) {
                const updatedBotResponses = message.botResponses.map((response) => ({
                    ...response,
                    isSelected: response.id === selectedResponseId, // Mark the selected response
                }));
                return { ...message, botResponses: updatedBotResponses };
            }
            return message;
        });

        setData({ messages: updatedMessages });
    };

    return (
        <div className={"h-full overflow-y-auto max-h-[540px] border border-gray-300 rounded-lg p-10 space-y-5 bg-[#0A0F24] text-white"} style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}>
            {messages.map((message) => {
                if (!message || !message.botResponses) {
                    console.error("Invalid message format:", message);
                    return null; // Skip rendering if message is invalid
                }

                return (
                    <div key={message.id} className="space-y-10">
                        <UserMessage text={message.text} />

                        {message.botResponses.map((botResponse) => {
                            // Only display the selected response or all responses if none are selected
                            if (botResponse.isSelected || message.botResponses.every((r) => !r.isSelected)) {
                                return (
                                    <BotMessage
                                        key={botResponse.id}
                                        text={botResponse.text}
                                        category={botResponse.categoryResponse?.data?.class || "Unknown"}
                                        intent={botResponse.intentResponse?.data?.class || "Unknown"}
                                        ner={"Unknown"}
                                        id={botResponse.id}
                                        isSelected={botResponse.isSelected}
                                        onSelect={() => handleSelectOption(message.id, botResponse.id)}
                                    />
                                );
                            }
                            return null;
                        })}
                    </div>
                );
            })}
        </div>
    );
};

export default MessageDisplay;