import { useHeaderContext } from "../context/HeaderContext"; // Import context
import TemporaryDrawer from "./TemporaryDrawer";

const Header = () => {
	return (
		<header
			className="py-4 text-white text-3xl font-bold tracking-widest flex items-center justify-center relative"
			style={{ backgroundColor: "black" }}
// Set background color with 79% opacity
		>
			<TemporaryDrawer className="absolute left-4" /> {/* Keeps the drawer button aligned to the left */}
		</header>
	);
};

export default Header;
