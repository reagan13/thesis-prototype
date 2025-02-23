import { useEffect, useRef, useMemo } from "react";
import UserMessage from "./UserMessage";
import BotMessage from "./BotMessage";
import PropTypes from "prop-types";
import { useData } from "../context/DataContext"; // Import useData
import { Link } from "react-router-dom";

const MessageDisplay = () => {
	const { data, setData, setSelectedBotResponse } = useData(); // Get setSelectedBotResponse from context
	const messages = useMemo(() => data.messages || [], [data.messages]); // Access messages from context
	const messageEndRef = useRef(null); // Create a ref for the message container

	// Handle selecting an option in bot responses
	const handleSelectOption = (messageId, selectedResponseId, botResponse) => {
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

		// Set the selected bot response in context
		setSelectedBotResponse(botResponse);
	};

	// Scroll to the bottom of the message container whenever messages change
	useEffect(() => {
		if (messageEndRef.current) {
			messageEndRef.current.scrollIntoView({ behavior: "smooth" });
		}
	}, [messages]);

	return (
		<div
			className="h-full overflow-y-auto space-y-10 pb-2"
			style={{ scrollbarWidth: "none" }} // Hide scrollbar for a cleaner look
		>
			{messages.map((message) => {
				if (!message || !message.botResponses) {
					console.error("Invalid message format:", message);
					return null; // Skip rendering if message is invalid
				}

				const selectedResponses = message.botResponses.filter(
					(botResponse) => botResponse.isSelected
				);

				const responsesToDisplay =
					selectedResponses.length > 0
						? selectedResponses
						: message.botResponses;

				return (
					<div key={message.id} className="space-y-4">
						{/* User Message */}
						<div className="flex justify-end">
							<UserMessage text={message.text} />
						</div>

						{/* Bot Responses */}
						<div
							className={`flex ${
								responsesToDisplay.length === 2
									? "justify-around"
									: "justify-start"
							}`}
						>
							{responsesToDisplay.map((botResponse) => (
								<Link key={botResponse.id} to={`/chat/${botResponse.id}`}>
									<BotMessage
										text={botResponse.text}
										category={
											botResponse.categoryResponse?.data?.class || "Unknown"
										}
										intent={
											botResponse.intentResponse?.data?.class || "Unknown"
										}
										ner={"Unknown"}
										id={botResponse.id}
										isSelected={botResponse.isSelected}
										onSelect={() =>
											handleSelectOption(
												message.id,
												botResponse.id,
												botResponse
											)
										}
									/>
								</Link>
							))}
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
			botResponses: PropTypes.arrayOf(
				PropTypes.shape({
					id: PropTypes.string.isRequired,
					text: PropTypes.string.isRequired,
					isSelected: PropTypes.bool,
					categoryResponse: PropTypes.object,
					intentResponse: PropTypes.object,
				})
			).isRequired,
		})
	),
};

export default MessageDisplay;