import React, { useState } from "react";
import { Paperclip, Send } from "lucide-react";

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
					{ text: `This AI chatbot has been developed to optimize communication and simplify work processes, ultimately leading to smoother operations.`, sender: "chatgpt" },
				]);
			}, 1000); // Simulate a delay for the response
		}
	};

	return (
		<div className="flex flex-col h-screen bg-[#034573] p-4">
			<div className="flex-1 overflow-y-auto p-4 space-y-4">
				{/* Messages display area */}
				{messages.map((msg, index) => (
					<div
						key={index}
						className={`flex items-center ${
							msg.sender === "user" ? "justify-end" : "justify-start"
						}`}
					>
						<div
							className={`p-3 max-w-[75%] rounded-lg shadow-md ${
								msg.sender === "user"
									? "bg-[#5BA4E5] text-white rounded-br-none"
									: "bg-[#A7D7F9] text-black rounded-bl-none"
							}`}
						>
							{msg.text}
						</div>
					</div>
				))}
			</div>

			{/* Input area */}
			<form
				onSubmit={handleSubmit}
				className="flex items-center w-full border-2 border-[#111852] rounded-2xl overflow-hidden bg-white focus-within:ring-2 focus-within:ring-[#111852]"
			>
				<input
					type="text"
					placeholder="Type a new message here"
					value={inputValue}
					onChange={(e) => setInputValue(e.target.value)}
					className="w-full px-4 py-3 text-gray-700 focus:outline-none"
				/>
				<button
					type="submit"
					className="p-3 text-[#111852] hover:text-blue-600 transition duration-200"
				>
					<Send size={20} />
				</button>
			</form>
		</div>
	);
};

export default Input;
