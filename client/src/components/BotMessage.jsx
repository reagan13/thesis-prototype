import PropTypes from "prop-types";
import { Link, useNavigate } from "react-router-dom";
import { useData } from "../context/DataContext";
import axios from "axios";

const BotMessage = ({
    text,
    timestamp,
    category,
    intent,
    ner,
    weightedSum,
    modelUsed,
    id,
    isSelected,
    onSelect,
}) => {
    const navigate = useNavigate();
    const {
        setSelectedChatAnalysis,
        data: { chatHistory, activeChatId },
    } = useData();

    const handleViewChatAnalysis = async (messageId) => {
        try {
            const activeChat = chatHistory.find((chat) => chat.id === activeChatId);
            if (!activeChat) throw new Error("Active chat not found.");

            const message = activeChat.messages.find(
                (msg) => msg.id === messageId || msg.botResponse?.id === messageId
            );
            if (!message) throw new Error(`Message with ID ${messageId} not found.`);

            const isUserMessage = message.id === messageId;
            const userMessage = isUserMessage
                ? message
                : activeChat.messages.find((msg) => msg.botResponse?.id === messageId);
            const botResponseMessage = isUserMessage
                ? activeChat.messages.find((msg) => msg.id === message.botResponse?.id)
                : message;

            if (!userMessage || !botResponseMessage) {
                throw new Error(
                    `Corresponding user message or bot response not found for message ID ${messageId}.`
                );
            }

            setSelectedChatAnalysis({
                chatId: activeChat.id,
                messageId: userMessage.id,
                text: userMessage.text,
                botResponse: botResponseMessage.botResponse,
            });

            navigate(`/chat-analysis/${userMessage.id}`);
        } catch (error) {
            console.error("Error navigating to Chat Analysis:", error);
            alert(`An error occurred: ${error.message}`);
        }
    };

    return (
        <div className="p-4 rounded-3xl space-y-4 text-left max-w-[500px] border border-gray-300 shadow-md bg-white">
            <div>
                <h3 className="text-lg font-semibold text-gray-800">Response:</h3>
                <p className="text-gray-700">{text}</p>
                <p className="text-xs text-gray-500">{timestamp}</p> {/* Display timestamp */}
            </div>
            <div className="space-y-2">
                <ul className="space-y-1 text-gray-700">
                    <li>
                        <span className="font-semibold">Model Used:</span>{" "}
                        {modelUsed || "Unknown"}
                    </li>
                </ul>
            </div>
            <div className="flex justify-end">
                <button
                    onClick={() => handleViewChatAnalysis(id)}
                    className="text-green-600 font-bold hover:text-green-800 transition duration-150"
                >
                    View Chat Analysis {">"}
                </button>
            </div>
        </div>
    );
};

BotMessage.propTypes = {
    text: PropTypes.string.isRequired,
    timestamp: PropTypes.string.isRequired,
    category: PropTypes.shape({
        label: PropTypes.string,
        prediction: PropTypes.string,
        confidence: PropTypes.number,
    }),
    intent: PropTypes.shape({
        label: PropTypes.string,
        prediction: PropTypes.string,
        confidence: PropTypes.number,
    }),
    ner: PropTypes.arrayOf(
        PropTypes.shape({
            label: PropTypes.string,
            entity: PropTypes.string,
        })
    ),
    weightedSum: PropTypes.number,
    modelUsed: PropTypes.string,
    id: PropTypes.number.isRequired,
    isSelected: PropTypes.bool.isRequired,
    onSelect: PropTypes.func.isRequired,
};

export default BotMessage;