import React, { useState } from "react";
import { Paperclip, Gift, Send } from "lucide-react"; // Importing Lucide icons

const Input = () => {
	const [inputValue, setInputValue] = useState(""); // State for input value
	const [messages, setMessages] = useState([]); // State for messages

	const handleSubmit = (e) => {
		e.preventDefault(); // Prevent default form submission
		if (inputValue.trim()) {
			// Only add non-empty messages
			setMessages([...messages, { text: inputValue, sender: "user" }]);
			setInputValue(""); // Clear input after submission

			// Simulate a response from ChatGPT
			setTimeout(() => {
				setMessages((prevMessages) => [
					...prevMessages,
					{ text: `Response to: ${inputValue}`, sender: "chatgpt" },
				]);
			}, 1000); // Simulate a delay for the response
		}
	};

	return (
		<div className="flex flex-col h-screen bg-gray-100">
			<h2 className="text-2xl font-semibold mb-4 text-center">
				What can I help with?
			</h2>
			<div className="flex-1 overflow-y-auto p-4">
				{/* Messages display area */}
				<div className="flex flex-col space-y-2">
					{messages.length === 0 ? (
						<p className="text-gray-500">
							No messages yet. Start the conversation!
						</p>
					) : (
						messages.map((msg, index) => (
							<div
								key={index}
								className={`p-2 rounded-lg ${
									msg.sender === "user"
										? "bg-blue-500 text-white self-end"
										: "bg-gray-200 text-black"
								}`}
							>
								{msg.text}
							</div>
						))
					)}
				</div>
			</div>

			{/* Input area */}
			<form
				onSubmit={handleSubmit}
				className="flex items-center p-2 bg-white border-t"
			>
				<input
					type="text"
					placeholder="Message ChatGPT"
					value={inputValue}
					onChange={(e) => setInputValue(e.target.value)}
					className="flex-grow border-none p-2 focus:outline-none rounded-l-lg h-12"
				/>
				<button
					type="submit"
					className="p-2 text-gray-500 hover:bg-gray-200 rounded-full h-12"
				>
					<Send size={20} />
				</button>
			</form>
		</div>
	);
};

export default Input;
