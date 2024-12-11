import { useState } from "react";
import { Send } from "lucide-react";
import PropTypes from "prop-types";
// MessageDisplay Component
const MessageDisplay = ({ messages }) => (
	<div className="h-full border overflow-y-auto max-h-[540px]">
		{messages.map((message) => (
			<div
				key={message.id}
				className={message.sender === "user" ? "text-right" : "text-left"}
			>
				{message.text}
			</div>
		))}
	</div>
);
MessageDisplay.propTypes = {
	messages: PropTypes.arrayOf(
		PropTypes.shape({
			id: PropTypes.number,
			text: PropTypes.string,
			sender: PropTypes.string,
		})
	),
};

// InputSection Component
const InputSection = ({ input, setInput, handleSend }) => (
	<div className="flex justify-between items-center gap-4">
		<input
			type="text"
			value={input}
			onChange={(e) => setInput(e.target.value)}
			placeholder="Enter your message..."
			className="rounded-full w-full border border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500 p-2"
		/>
		<button
			onClick={handleSend}
			className="bg-blue-500 text-darkNavy rounded-full p-2 hover:bg-blue-600 transition duration-200"
		>
			<Send />
		</button>
	</div>
);
InputSection.propTypes = {
	input: PropTypes.string,
	setInput: PropTypes.func,
	handleSend: PropTypes.func,
};

const ChatInterface = () => {
	const [messages, setMessages] = useState([]);
	const [input, setInput] = useState("");

	const handleSend = () => {
		if (input.trim()) {
			const newMessage = {
				id: messages.length + 1,
				text: input,
				sender: "user",
			};
			setMessages((prevMessages) => [...prevMessages, newMessage]);
			setInput("");

			// Simulate a response from ChatGPT
			const botResponse = {
				id: messages.length + 2,
				text: `This is a simulated response to: "${input}"`,
				sender: "bot",
			};
			setMessages((prevMessages) => [...prevMessages, botResponse]);
		}
	};

	return (
		<>
			{/* Conditional rendering based on messages */}
			{messages.length === 0 ? (
				<div className="border border-gray-300 rounded-lg shadow-lg w-[700px] p-6 bg-white">
					<div className="space-y-6 text-center">
						<h1 className="text-3xl font-extrabold tracking-wide text-gray-800">
							What Can I Help With?
						</h1>
						<InputSection
							input={input}
							setInput={setInput}
							handleSend={handleSend}
						/>
					</div>
				</div>
			) : (
				<div className="border border-gray-300 rounded-lg shadow-lg w-[700px] p-6 bg-white h-full">
					<div className="space-y-6 text-center justify-between flex flex-col h-full">
						<MessageDisplay messages={messages} />
						<InputSection
							input={input}
							setInput={setInput}
							handleSend={handleSend}
						/>
					</div>
				</div>
			)}
		</>
	);
};

export default ChatInterface;
