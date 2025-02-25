import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { Menu, MessageSquare, Trash2, Plus } from "lucide-react"; // Import Lucide icons
import { motion, AnimatePresence } from "framer-motion";
import { useData } from "../context/DataContext";

// Utility function to truncate text
const truncateText = (text, maxLength = 20) => {
	if (!text) return "New Chat"; // Default title if no text is provided
	return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
};

export default function Sidebar() {
	const navigate = useNavigate();
	const { data, setData, setIsSidebarCollapsed, isSidebarCollapsed } =
		useData();

	// Toggle sidebar collapse
	const toggleSidebar = () => {
		setIsSidebarCollapsed((prev) => !prev);
	};

	// Delete a specific chat
	const handleDeleteChat = (id) => {
		const updatedChatHistory = data.chatHistory.filter(
			(chat) => chat.id !== id
		);

		// Update global state
		setData({
			...data,
			chatHistory: updatedChatHistory,
			activeChatId:
				data.activeChatId === id && updatedChatHistory.length > 0
					? updatedChatHistory[0].id // Set the first chat as active if the current one is deleted
					: null,
		});

		if (data.activeChatId === id) {
			navigate("/chat"); // Navigate to the latest chat if the active chat is deleted
		}

		alert(`Chat ${id} deleted!`);
	};

	// Start a new chat
	const handleNewChat = () => {
		const newChatId = Date.now(); // Generate a unique ID
		const newChat = {
			id: newChatId,
			title: "New Chat", // Default title for new chats
			messages: [],
		};

		// Update global state
		setData({
			...data,
			chatHistory: [newChat, ...data.chatHistory], // Add new chat to the top
			activeChatId: newChatId, // Set the new chat as active
		});

		navigate(`/chat/${newChatId}`); // Navigate to the new chat
	};

	// Switch to a specific chat
	const handleSwitchChat = (id) => {
		// Update global state
		setData({
			...data,
			activeChatId: id, // Set the clicked chat as active
		});

		navigate(`/chat/${id}`); // Navigate to the chat
	};

	return (
		<motion.div
			key={isSidebarCollapsed ? "collapsed" : "expanded"}
			className="flex flex-col h-full bg-white shadow-lg"
			initial={false}
			animate={{ width: isSidebarCollapsed ? "80px" : "270px" }}
			transition={{ duration: 0.3, ease: "easeInOut" }}
		>
			{/* Divider */}
			<hr className="border-gray-300 mx-4" />

			{/* Main Content */}
			<motion.div
				className="flex-1 overflow-y-auto"
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ duration: 0.3 }}
			>
				<ul className="space-y-2 px-4 py-4">
					{/* CHATTIBOT Title and Menu Button Grouped Together */}
					<li className="flex items-center gap-2 p-2 rounded-md hover:bg-gray-200 transition-all duration-300 mb-5">
						<button
							className="flex items-center gap-2 w-full"
							onClick={toggleSidebar}
						>
							<Menu size={30} />
							<AnimatePresence>
								{!isSidebarCollapsed && (
									<motion.h1
										key="chattibot-title"
										initial={{ opacity: 0 }}
										animate={{ opacity: 1 }}
										exit={{ opacity: 0 }}
										transition={{ duration: 0.2 }}
										className="text-2xl font-bold tracking-widest"
									>
										CHATTIBOT
									</motion.h1>
								)}
							</AnimatePresence>
						</button>
					</li>

					{/* New Chat Button */}
					<li>
						<button
							onClick={handleNewChat}
							className="flex items-center gap-2 w-full p-2 text-green-600 bg-green-100 rounded-md hover:bg-green-200 transition-all duration-300"
						>
							<Plus size={20} />
							<AnimatePresence>
								{!isSidebarCollapsed && (
									<motion.span
										key="new-chat-text"
										initial={{ opacity: 0 }}
										animate={{ opacity: 1 }}
										exit={{ opacity: 0 }}
										transition={{ duration: 0.2 }}
									>
										New Chat
									</motion.span>
								)}
							</AnimatePresence>
						</button>
					</li>

					{/* Chat History Section */}
					<li>
						<div className="mt-4">
							<AnimatePresence>
								{!isSidebarCollapsed && (
									<motion.h2
										key="chat-history-title"
										initial={{ opacity: 0 }}
										animate={{ opacity: 1 }}
										exit={{ opacity: 0 }}
										transition={{ duration: 0.2 }}
										className="text-lg font-semibold text-gray-700 mb-2"
									>
										Chat History
									</motion.h2>
								)}
							</AnimatePresence>
							<ul className="space-y-2">
								{data.chatHistory.length > 0 ? (
									data.chatHistory.map((chat) => {
										// Determine the chat title based on the first user message
										const firstUserMessage =
											chat.messages.find((msg) => msg.sender === "user")
												?.text || "New Chat";

										// Truncate the chat title if it's too long
										const truncatedTitle = truncateText(firstUserMessage);

										return (
											<li key={chat.id}>
												<div className="flex items-center justify-between gap-2 p-2 rounded-md hover:bg-gray-200 transition-all duration-300">
													<button
														onClick={() => handleSwitchChat(chat.id)}
														className="flex items-center gap-2 flex-grow max-w-44"
													>
														<MessageSquare size={20} />
														<AnimatePresence>
															{!isSidebarCollapsed && (
																<motion.span
																	key={`chat-${chat.id}`}
																	initial={{ opacity: 0 }}
																	animate={{ opacity: 1 }}
																	exit={{ opacity: 0 }}
																	transition={{ duration: 0.2 }}
																	className="truncate"
																>
																	{truncatedTitle}
																</motion.span>
															)}
														</AnimatePresence>
													</button>
													<button
														onClick={() => handleDeleteChat(chat.id)}
														className="text-red-600 hover:text-red-800 transition-colors"
													>
														<Trash2 size={16} />
													</button>
												</div>
											</li>
										);
									})
								) : (
									<p className="text-sm text-gray-500">No chat history</p>
								)}
							</ul>
						</div>
					</li>
				</ul>
			</motion.div>
		</motion.div>
	);
}
