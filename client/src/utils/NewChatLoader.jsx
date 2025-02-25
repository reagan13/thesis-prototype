// utils/NewChatLoader.jsx
import { Navigate } from "react-router-dom";
import { useData } from "../context/DataContext"; // Adjust the path as needed

const NewChatLoader = () => {
	const { setData } = useData();

	const newChatId = Date.now();
	const newChat = {
		id: newChatId,
		title: `Conversation ${newChatId}`,
		messages: [],
	};

	// Update global state with the new chat
	setData((prevData) => ({
		...prevData,
		chatHistory: [newChat, ...prevData.chatHistory],
		activeChatId: newChatId,
	}));

	return <Navigate to={`/chat/${newChatId}`} replace />;
};

export default NewChatLoader;
