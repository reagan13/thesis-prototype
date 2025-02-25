import { useEffect, useRef, useMemo } from "react";
import UserMessage from "./UserMessage";
import BotMessage from "./BotMessage";
import PropTypes from "prop-types";
import { useData } from "../context/DataContext";

const MessageDisplay = () => {
	const { data } = useData();
	const activeChat = data.chatHistory.find(
		(chat) => chat.id === data.activeChatId
	);
	const messages = useMemo(() => activeChat?.messages || [], [activeChat]);

	const messageEndRef = useRef(null);

	useEffect(() => {
		if (messageEndRef.current) {
			messageEndRef.current.scrollIntoView({ behavior: "smooth" });
		}
	}, [messages]);

	return (
		<div
			className="h-full overflow-y-auto space-y-10 pb-2"
			style={{ scrollbarWidth: "none" }}
		>
			{messages.length === 0 && (
				<div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
					<p className="text-lg font-semibold">Welcome to CHATTIBOT!</p>
					<p className="text-sm">
						Start a conversation by typing your message below.
					</p>
				</div>
			)}

			{messages.map((message, index) => {
				if (!message || !message.botResponse) {
					console.error("Invalid message format:", message);
					return null;
				}

				return (
					<div key={`message-${index}`} className="space-y-4">
						{/* User Message */}
						<div className="flex justify-end">
							<UserMessage text={message.text} />
						</div>
						{/* Bot Response */}
						<div className="flex justify-start">
							<BotMessage
								key={`bot-response-${index}`}
								text={message.botResponse.text}
								category={message.botResponse.predictions?.category} // Nested under predictions
								intent={message.botResponse.predictions?.intent} // Nested under predictions
								ner={message.botResponse.predictions?.ner || []} // Default to empty array
								weightedSum={message.botResponse.weightedSum}
								modelUsed={message.botResponse.modelUsed}
								id={message.botResponse.id}
								isSelected={false}
								onSelect={() => {}}
							/>
						</div>
					</div>
				);
			})}
			<div ref={messageEndRef}></div>
		</div>
	);
};

MessageDisplay.propTypes = {
	messages: PropTypes.arrayOf(
		PropTypes.shape({
			id: PropTypes.string.isRequired,
			text: PropTypes.string.isRequired,
			botResponse: PropTypes.shape({
				id: PropTypes.string.isRequired,
				text: PropTypes.string.isRequired,
			}).isRequired,
		})
	),
};

export default MessageDisplay;
