import React from "react";
import { motion } from "framer-motion";

const LoadingLogo = () => {
	return (
		<div className="fixed inset-0 flex flex-col items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm z-50">
			<div className="relative mt-4">
				<img src="../../public/logo3.PNG" alt="Logo" className="w-60 h-60" />
				<div className="absolute inset-0 flex items-center justify-center">
					<motion.svg
						className="w-64 h-64"
						viewBox="0 0 100 100"
						animate={{ rotate: 360 }}
						transition={{ repeat: Infinity, duration: 5, ease: "linear" }}
					>
						<circle
							cx="50"
							cy="50"
							r="45"
							stroke="white"
							strokeWidth="5"
							fill="none"
							strokeLinecap="round"
							strokeDasharray="31.4, 31.4"
							strokeDashoffset="0"
						/>
					</motion.svg>
				</div>
			</div>
		</div>
	);
};

export default LoadingLogo;
