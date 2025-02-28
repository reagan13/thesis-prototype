import PropTypes from "prop-types";
import { useNavigate } from "react-router-dom";
import { useData } from "../context/DataContext";

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
		<div className={`flex items-end ${isSelected ? "bg-gray-100" : ""}`}>
			{/* Bot Avatar (left side) */}
			<img
				src={"../../public/logo.PNG"}
				alt="Bot Avatar"
				className="w-14 h-14 rounded-full mr-3"
			/>

			<div className="flex flex-col max-w-[700px]">
				{/* Message Bubble */}
				<div
					className={`relative p-4 border border-black ${
						text.length > 100
							? "rounded-r-3xl rounded-tl-3xl rounded-bl-none"
							: "rounded-r-full rounded-tl-full rounded-bl-none"
					}`}
				>
					<p className="whitespace-pre-wrap">{text}</p>
					<div className="flex justify-between items-center pt-4 gap-10">
						<p>
							<span className="font-bold">Model:</span> {modelUsed}
						</p>
						<button
							onClick={() => handleViewChatAnalysis(id)}
							className="font-semibold py-2 px-4 rounded-full bg-black text-white hover:border-black hover:border hover:bg-white hover:text-black "
						>
							Chat Analysis {">"}
						</button>
					</div>
				</div>
				{/* Timestamp */}
				<p className="text-sm text-gray-500 mt-1">{timestamp}</p>
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
