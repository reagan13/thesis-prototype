const Footer = () => {
	return (
		<div className="border-t border-gray-300 p-4 h-24 mx-48 flex items-center justify-center">
			<input
				type="text"
				placeholder="Type your message..."
				className="border border-gray-300 rounded-l px-4 py-2 w-full"
			/>
			<button className="bg-blue-500 text-white px-4 py-2 rounded-r">
				Submit
			</button>
		</div>
	);
};

export default Footer;
