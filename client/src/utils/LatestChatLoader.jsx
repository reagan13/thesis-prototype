// utils/LatestChatLoader.jsx
import { Navigate } from "react-router-dom";
import { useData } from "../context/DataContext"; // Adjust the path as needed

const LatestChatLoader = () => {
	const { data } = useData();
	const latestChatId = data.chatHistory[0]?.id;

	if (!latestChatId) {
		// Redirect to create a new chat if no chats exist
		return <Navigate to="/chat/new" replace />;
	}

	return <Navigate to={`/chat/${latestChatId}`} replace />;
};

export default LatestChatLoader;
