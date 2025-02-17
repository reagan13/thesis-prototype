import SearchInput from "./SearchInput";
const Footer = () => {
	return (
		<div className="border-t border-gray-300 p-4 h-16 flex flex-col md:flex-row items-center justify-center space-y-2 md:space-y-0 md:space-x-2">
			<SearchInput />
		</div>
	);
};

export default Footer;
