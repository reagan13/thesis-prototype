import PropTypes from "prop-types";
import { Link, useNavigate } from "react-router-dom";
import { useData } from "../context/DataContext";
import axios from "axios";

const BotMessage = ({
	text,
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
		data: { chatHistory, activeChatId }, // Access chatHistory and activeChatId from DataContext
	} = useData();

	const handleViewChatAnalysis = async (messageId) => {
		try {
			console.log("Navigating to Chat Analysis for message ID:", messageId);

			// Log the chat history and active chat ID for debugging
			console.log("Chat History:", chatHistory);
			console.log("Active Chat ID:", activeChatId);

			// Find the active chat
			const activeChat = chatHistory.find((chat) => {
				console.log(`Chat ID: ${chat.id}, Active Chat ID: ${activeChatId}`);
				return chat.id === activeChatId;
			});
			if (!activeChat) {
				throw new Error("Active chat not found.");
			}

			// Log the messages in the active chat
			activeChat.messages.forEach((msg) => {
				console.log(
					`Message ID: ${msg.id}, Bot Response ID: ${msg.botResponse?.id}`
				);
			});

			// Find the message (either user or bot response) by messageId
			const message = activeChat.messages.find(
				(msg) => msg.id === messageId || msg.botResponse?.id === messageId
			);
			if (!message) {
				throw new Error(`Message with ID ${messageId} not found.`);
			}

			// Determine if the found message is a user message or a bot response
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

			// Set the selected chat analysis data in the global state
			setSelectedChatAnalysis({
				chatId: activeChat.id,
				messageId: userMessage.id,
				text: userMessage.text, // User input
				botResponse: botResponseMessage.botResponse, // Bot response data
			});

			// Navigate to the Chat Analysis page with the messageId as a route parameter
			navigate(`/chat-analysis/${userMessage.id}`);
		} catch (error) {
			console.error("Error navigating to Chat Analysis:", error);
			alert(`An error occurred: ${error.message}`);
		}
	};
	return (
		<div className="p-4 rounded-3xl space-y-4 text-left max-w-[500px] border border-gray-300 shadow-md bg-white">
			{/* Bot Response */}
			<div>
				<h3 className="text-lg font-semibold text-gray-800">Response:</h3>
				<p className="text-gray-700">{text}</p>{" "}
				{/* Display only the generated text */}
			</div>
			{/* Metadata Section */}
			<div className="space-y-2">
				{/* <h4 className="text-sm font-medium text-gray-600">Metadata:</h4> */}
				<ul className="space-y-1 text-gray-700">
					{/* Model Used */}
					<li>
						<span className="font-semibold">Model Used:</span>{" "}
						{modelUsed || "Unknown"}
					</li>
					{/* Category with Confidence */}
					{/* <li>
						<span className="font-semibold">Category:</span>{" "}
						{category?.label || category?.prediction || "Unknown"}{" "}
						<span className="text-xs text-gray-500">
							(Confidence:{" "}
							{category?.confidence
								? `${(category.confidence * 100).toFixed(2)}%`
								: "N/A"}
							)
						</span>
					</li> */}
					{/* Intent with Confidence */}
					{/* <li>
						<span className="font-semibold">Intent:</span>{" "}
						{intent?.label || intent?.prediction || "Unknown"}{" "}
						<span className="text-xs text-gray-500">
							(Confidence:{" "}
							{intent?.confidence
								? `${(intent.confidence * 100).toFixed(2)}%`
								: "N/A"}
							)
						</span>
					</li> */}
					{/* Named Entity Recognition (NER) */}
					{/* <li>
						<span className="font-semibold">NER:</span>{" "}
						{Array.isArray(ner) && ner.length > 0 ? (
							ner.map((entity, index) => (
								<span key={index} className="mr-2">
									{entity.label}: {entity.entity}
								</span>
							))
						) : (
							<span>Unknown</span>
						)}
					</li> */}
					{/* Weighted Sum */}
					{/* <li>
						<span className="font-semibold">Weighted Sum:</span>{" "}
						{weightedSum !== undefined ? weightedSum.toFixed(4) : "N/A"}
					</li> */}
				</ul>
			</div>
			{/* Action Button */}
			<div className="flex justify-end">
				<button
					onClick={() => handleViewChatAnalysis(id)} // Pass the message ID here
					className="text-green-600 font-bold hover:text-green-800 transition duration-150"
				>
					View Chat Analysis {">"}
				</button>
			</div>
		</div>
	);
};

// Define PropTypes for type-checking
BotMessage.propTypes = {
	text: PropTypes.string.isRequired,
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
	modelUsed: PropTypes.string, // Add modelUsed prop
	id: PropTypes.number.isRequired,
	isSelected: PropTypes.bool.isRequired,
	onSelect: PropTypes.func.isRequired,
};

export default BotMessage;
