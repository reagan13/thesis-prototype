import { useState } from "react";

const LeftPage = () => {
	const [input, setInput] = useState("");
	const [messages, setMessages] = useState([]);

	const responses = [
		"Response 1: This is a random response.",
		"Response 2: Here is another random response.",
		"Response 3: Yet another random response.",
	];

	const handleSubmit = (e) => {
		e.preventDefault();
		if (input.trim() === "") return;

		const randomResponse =
			responses[Math.floor(Math.random() * responses.length)];
		setMessages([
			...messages,
			{ type: "user", text: input },
			{ type: "bot", text: randomResponse },
		]);
		setInput("");
	};

	return (
		<div className="p-4 h-full flex flex-col">
			<div className="flex-grow overflow-y-auto mb-4">
				{messages.map((message, index) => (
					<div
						key={index}
						className={`mb-2 ${
							message.type === "user" ? "text-right" : "text-left"
						}`}
					>
						<div
							className={`inline-block p-2 rounded ${
								message.type === "user"
									? "bg-blue-500 text-white"
									: "bg-gray-200 text-black"
							}`}
						>
							{message.text}
						</div>
					</div>
				))}
			</div>
			<form onSubmit={handleSubmit} className="flex">
				<input
					type="text"
					placeholder="Type your message..."
					value={input}
					onChange={(e) => setInput(e.target.value)}
					className="border border-gray-300 rounded-l px-4 py-2 w-full"
				/>
				<button
					type="submit"
					className="bg-blue-500 text-white px-4 py-2 rounded-r"
				>
					Submit
				</button>
			</form>
		</div>
	);
};

export default LeftPage;
