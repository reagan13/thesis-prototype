import { useEffect } from "react";
import UserMessage from "./UserMessage";
import BotMessage from "./BotMessage";
import PropTypes from "prop-types";
import { useData } from "../context/DataContext"; // Import useData

const MessageDisplay = () => {
	const { data } = useData(); // Get data from context
	const messages = data.messages || []; // Access messages from context

	return (
		<div className="h-full overflow-y-auto max-h-[540px] border border-gray-300 rounded-lg p-10 space-y-5">
			{messages.map((message) => {
				console.log(message);
				console.log(message.botResponse.categoryResponse.data.class); // Log the message object
				console.log(message.botResponse.intentResponse.data.class); // Log the message object
				console.log("bot response id", message.botResponse.id); // Log the message object
				return (
					<div key={message.id}>
						<UserMessage text={message.text} />

						<BotMessage
							text={message.botResponse.text} // Accessing bot response text
							category={message.botResponse.categoryResponse.data.class} // Adjust based on your data structure
							intent={message.botResponse.intentResponse.data.class} // Adjust based on your data structure
							ner={"Unknown"}
							id={message.botResponse.id}
						/>
					</div>
				);
			})}
		</div>
	);
};

export default MessageDisplay;
