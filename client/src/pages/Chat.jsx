import React from "react";
import ChatInterface from "../components/ChatInterface";
import { useData } from "../context/DataContext";

const Chat = () => {
	const { data } = useData();
	const activeChat = data.chatHistory.find(
		(chat) => chat.id === data.activeChatId
	);

	if (!activeChat) {
		return (
			<div className="flex justify-center items-center h-screen">
				Chat not found!
			</div>
		);
	}

	return <ChatInterface />;
};

export default Chat;
