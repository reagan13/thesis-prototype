import React, { useState } from "react";
import PropTypes from "prop-types";
import { motion } from "framer-motion";
import {
	Home,
	Settings,
	Users,
	LogOut,
	ChevronLeft,
	ChevronRight,
} from "lucide-react";

const SideBar = () => {
	const [isCollapsed, setIsCollapsed] = useState(false);

	const toggleSidebar = () => {
		setIsCollapsed(!isCollapsed);
	};

	return (
		<motion.aside
			className={`bg-black text-white h-full flex flex-col justify-between ${
				isCollapsed ? "w-20" : "w-64"
			} transition-all duration-300 ease-in-out`}
			initial={{ width: 256 }}
			animate={{ width: isCollapsed ? 80 : 256 }}
		>
			{/* Top Section */}
			<div className="p-4">
				<div className="flex items-center justify-between mb-6">
					{!isCollapsed && <h1 className="text-lg font-bold">Dashboard</h1>}
					<button onClick={toggleSidebar}>
						{isCollapsed ? (
							<ChevronRight size={20} />
						) : (
							<ChevronLeft size={20} />
						)}
					</button>
				</div>

				{/* Navigation Links */}
				<nav className="space-y-4">
					<NavItem
						icon={<Home size={20} />}
						label="Home"
						collapsed={isCollapsed}
					/>
					<NavItem
						icon={<Users size={20} />}
						label="Users"
						collapsed={isCollapsed}
					/>
					<NavItem
						icon={<Settings size={20} />}
						label="Settings"
						collapsed={isCollapsed}
					/>
				</nav>
			</div>

			{/* Bottom Section */}
			<div className="p-4 border-t border-gray-700">
				<NavItem
					icon={<LogOut size={20} />}
					label="Logout"
					collapsed={isCollapsed}
				/>
			</div>
		</motion.aside>
	);
};

// NavItem Component
const NavItem = ({ icon, label, collapsed }) => {
	return (
		<div className="flex items-center space-x-4">
			{icon}
			{!collapsed && <span>{label}</span>}
		</div>
	);
};
NavItem.propTypes = {
	icon: PropTypes.element.isRequired,
	label: PropTypes.string.isRequired,
	collapsed: PropTypes.bool.isRequired,
};

export default SideBar;
