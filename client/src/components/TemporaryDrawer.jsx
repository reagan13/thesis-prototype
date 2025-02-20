import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { Menu, Home, BarChart, Trash2 } from "lucide-react"; // Import Lucide icons
import { motion, AnimatePresence } from "framer-motion";
import { useData } from "../context/DataContext";

export default function Sidebar() {
	const navigate = useNavigate();
	const { setData, isSidebarCollapsed, setIsSidebarCollapsed } = useData(); // Get state from context

	// Toggle sidebar collapse
	const toggleSidebar = () => {
		setIsSidebarCollapsed((prev) => !prev); // Toggle state
	};

	// Clear local storage and reset data
	const handleDeleteStorage = () => {
		localStorage.removeItem("chatData");
		setData({ messages: [] });
		alert("Local storage cleared!");
		navigate("/home");
	};

	return (
		<>
			<motion.div
				key={isSidebarCollapsed ? "collapsed" : "expanded"} // Ensures re-render on state change
				className="flex flex-col h-full"
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

						{/* Home Link */}
						<li>
							<Link
								to="/home"
								className="flex items-center gap-2 p-2 rounded-md hover:bg-gray-200 transition-all duration-300"
							>
								<Home size={20} />
								<AnimatePresence>
									{!isSidebarCollapsed && (
										<motion.span
											key="home-text"
											initial={{ opacity: 0 }}
											animate={{ opacity: 1 }}
											exit={{ opacity: 0 }}
											transition={{ duration: 0.2 }}
										>
											Home
										</motion.span>
									)}
								</AnimatePresence>
							</Link>
						</li>

						{/* Results Link */}
						<li>
							<Link
								to="/results"
								className="flex items-center gap-2 p-2 rounded-md hover:bg-gray-200 transition-all duration-300"
							>
								<Home size={20} />
								<AnimatePresence>
									{!isSidebarCollapsed && (
										<motion.span
											key="home-text"
											initial={{ opacity: 0 }}
											animate={{ opacity: 1 }}
											exit={{ opacity: 0 }}
											transition={{ duration: 0.2 }}
										>
											Results
										</motion.span>
									)}
								</AnimatePresence>
							</Link>
						</li>
					</ul>

					{/* Divider */}
					<hr className="border-gray-300 mx-4" />
					<ul className="space-y-2 px-4 py-2">
						{/* CHATTIBOT Title and Menu Button Grouped Together */}
						<li className="flex items-center gap-2 p-2 rounded-md  transition-all duration-300 ">
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
										Code
									</motion.h1>
								)}
							</AnimatePresence>
						</li>

						{/* Home Link */}
						<li>
							<Link
								to="/home"
								className="flex items-center gap-2 p-2 rounded-md hover:bg-gray-200 transition-all duration-300"
							>
								<Home size={20} />
								<AnimatePresence>
									{!isSidebarCollapsed && (
										<motion.span
											key="home-text"
											initial={{ opacity: 0 }}
											animate={{ opacity: 1 }}
											exit={{ opacity: 0 }}
											transition={{ duration: 0.2 }}
										>
											Home
										</motion.span>
									)}
								</AnimatePresence>
							</Link>
						</li>

						{/* Results Link */}
						<li>
							<Link
								to="/result"
								className="flex items-center gap-2 p-2 rounded-md hover:bg-gray-200 transition-all duration-300"
							>
								<BarChart size={20} />
								<AnimatePresence>
									{!isSidebarCollapsed && (
										<motion.span
											key="results-text"
											initial={{ opacity: 0 }}
											animate={{ opacity: 1 }}
											exit={{ opacity: 0 }}
											transition={{ duration: 0.2 }}
										>
											Results
										</motion.span>
									)}
								</AnimatePresence>
							</Link>
						</li>
					</ul>
					{/* Divider */}
					<hr className="border-gray-300 mx-4" />
					<ul className="space-y-2 px-4 py-2">
						{/* CHATTIBOT Title and Menu Button Grouped Together */}
						<li className="flex items-center gap-2 p-2 rounded-md  transition-all duration-300 ">
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
										Results
									</motion.h1>
								)}
							</AnimatePresence>
						</li>

						{/* Home Link */}
						<li>
							<Link
								to="/home"
								className="flex items-center gap-2 p-2 rounded-md hover:bg-gray-200 transition-all duration-300"
							>
								<Home size={20} />
								<AnimatePresence>
									{!isSidebarCollapsed && (
										<motion.span
											key="home-text"
											initial={{ opacity: 0 }}
											animate={{ opacity: 1 }}
											exit={{ opacity: 0 }}
											transition={{ duration: 0.2 }}
										>
											Home
										</motion.span>
									)}
								</AnimatePresence>
							</Link>
						</li>

						{/* Results Link */}
						<li>
							<Link
								to="/result"
								className="flex items-center gap-2 p-2 rounded-md hover:bg-gray-200 transition-all duration-300"
							>
								<BarChart size={20} />
								<AnimatePresence>
									{!isSidebarCollapsed && (
										<motion.span
											key="results-text"
											initial={{ opacity: 0 }}
											animate={{ opacity: 1 }}
											exit={{ opacity: 0 }}
											transition={{ duration: 0.2 }}
										>
											Results
										</motion.span>
									)}
								</AnimatePresence>
							</Link>
						</li>
					</ul>
					{/* Divider */}
					<hr className="border-gray-300 mx-4" />

					{/* Delete Storage Button */}
					<ul className="space-y-2 px-4 py-4">
						<li>
							<button
								onClick={handleDeleteStorage}
								className="flex items-center gap-2 w-full p-2 rounded-md bg-gray-200 hover:bg-gray-300 transition-all duration-300"
							>
								<Trash2 size={20} />
								<AnimatePresence>
									{!isSidebarCollapsed && (
										<motion.span
											key="delete-text"
											initial={{ opacity: 0 }}
											animate={{ opacity: 1 }}
											exit={{ opacity: 0 }}
											transition={{ duration: 0.2 }}
										>
											Delete Storage
										</motion.span>
									)}
								</AnimatePresence>
							</button>
						</li>
					</ul>
				</motion.div>
			</motion.div>
		</>
	);
}
