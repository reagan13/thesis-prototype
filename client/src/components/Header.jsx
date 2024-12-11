import { useHeaderContext } from "../context/HeaderContext"; // Import context
import TemporaryDrawer from "./TemporaryDrawer";

const Header = () => {
	const { headerName } = useHeaderContext(); // Use context to get headerName

	return (
		<header className=" py-4 text-left flex items-center text-3xl font-bold tracking-widest relative">
			<TemporaryDrawer />

			{headerName}
		</header>
	);
};

export default Header;
