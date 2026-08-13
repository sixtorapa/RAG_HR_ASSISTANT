"""User management CLI for the Flask app.

Keeps backward compatibility with the old behavior:
  python create_admin.py
will ensure an admin user exists.

Now it also supports creating/updating any user:
  python create_admin.py --username alice --role user
  python create_admin.py --username bob --password "S3cret!" --role admin --reset-password

Bulk mode (JSON file with a list of users):
  python create_admin.py --users-file ./users.json --reset-password

Example users.json:
[
  {"username": "alice", "password": "...", "role": "user", "is_active": true},
  {"username": "bob",   "password": "...", "role": "admin"}
]
"""


# Create a user (it will prompt for the password if you do not pass one):

# python create_admin.py --username valeria --role user

# Create a user with an inline password:

# python create_admin.py --username valeria --password "MiPass123!" --role user

# Reset the password of an existing user:

# python create_admin.py --username valeria --password "NuevaPass123!" --reset-password

# Create several users from JSON:

# python create_admin.py --users-file users.json --reset-password

# Generar password aleatoria:

# python create_admin.py --username valeria --role user --generate-password

import argparse
import json
import secrets
import sys
from getpass import getpass
from typing import Any, Dict, Iterable, Optional

from app import create_app, db
from app.models import User


def _bool_from_any(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _generate_password(length: int = 18) -> str:
    # urlsafe already avoids problematic characters for copying.
    return secrets.token_urlsafe(max(12, length))


def _departments_from_any(v: Any) -> Optional[list]:
    """
    Accepts a list (from the bulk JSON) or a "dept_a,dept_b" string (from the CLI).
    None -> nothing was specified (the user's existing field is left untouched).
    """
    if v is None:
        return None
    if isinstance(v, list):
        return [str(d).strip() for d in v if str(d).strip()]
    s = str(v).strip()
    if not s:
        return []
    return [d.strip() for d in s.split(",") if d.strip()]


def upsert_user(
    *,
    username: str,
    password: Optional[str],
    role: str,
    is_active: bool,
    email: Optional[str] = None,
    reset_password: bool = False,
    allowed_departments: Optional[list] = None,
) -> Dict[str, Any]:
    """Create user if missing; update basic fields; optionally reset password.

    allowed_departments (guardarril de acceso, ver User.get_allowed_departments):
    None = leave the existing value alone (or empty for a new user) ->
    fail closed por defecto para usuarios "user". Para "admin" no aplica (siempre
    acceso total, independientemente de este campo).
    """
    username = (username or "").strip()
    if not username:
        raise ValueError("username is required")

    if role not in {"user", "admin"}:
        raise ValueError("role must be 'user' or 'admin'")

    user = User.query.filter_by(username=username).first()
    created = False

    if not user:
        user = User(username=username)
        db.session.add(user)
        created = True

    # Always keep these in sync with the requested state
    user.role = role
    user.is_active = bool(is_active)
    if email:
        user.email = email
    if allowed_departments is not None:
        user.allowed_departments = allowed_departments

    if created or reset_password:
        if not password:
            raise ValueError(f"password required for user '{username}' (or use --generate-password)")
        user.set_password(password)

    db.session.commit()
    return {
        "username": username,
        "created": created,
        "role": role,
        "is_active": is_active,
        "allowed_departments": user.allowed_departments,
    }


def _load_users_file(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("users-file JSON must be a list")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each user item in users-file must be an object")
        yield item


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Create/update users for the app")

    parser.add_argument("--username", help="Username to create/update")
    parser.add_argument("--password", help="Password to set (use with --reset-password or for new user)")
    parser.add_argument("--email", help="Email (optional)")
    parser.add_argument("--role", default=None, choices=["user", "admin"], help="Role")
    parser.add_argument(
        "--inactive",
        action="store_true",
        help="Create/update user as inactive (default: active)",
    )
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="Reset password if user already exists",
    )
    parser.add_argument(
        "--generate-password",
        action="store_true",
        help="Generate a random password (printed once)",
    )
    parser.add_argument(
        "--allowed-departments",
        default=None,
        help=(
            "Comma-separated list of department slugs this user can access "
            "(e.g. 'compensation_benefits,recruitment_talent'). Ignored for role=admin "
            "(admins always have full access). Omitting this leaves a 'user' with no "
            "department access at all (fail closed) unless already set."
        ),
    )
    parser.add_argument(
        "--users-file",
        help="Path to JSON file with a list of users for bulk creation/update",
    )

    args = parser.parse_args(argv)

    app = create_app()
    with app.app_context():
        # Backward compatible mode: ensure admin exists.
        if not args.username and not args.users_file:
            username = "admin"
            password = "admin1234"
            user = User.query.filter_by(username=username).first()
            if not user:
                user = User(username=username, role="admin", is_active=True)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                print("✅ Admin creado:", username, password)
            else:
                print("ℹ️ Admin ya existe:", username)
            return 0

        # Bulk mode
        if args.users_file:
            results = []
            for u in _load_users_file(args.users_file):
                username = (u.get("username") or "").strip()
                role = (u.get("role") or args.role or "user").strip()
                is_active = _bool_from_any(u.get("is_active"), default=not args.inactive)
                email = u.get("email") or None

                password = u.get("password") or args.password
                generated_password = None
                if not password and args.generate_password:
                    generated_password = _generate_password()
                    password = generated_password
                if not password:
                    raise ValueError(
                        f"Missing password for '{username}'. Provide it in JSON or use --generate-password"
                    )

                res = upsert_user(
                    username=username,
                    password=password,
                    role=role,
                    is_active=is_active,
                    email=email,
                    reset_password=args.reset_password,
                    allowed_departments=_departments_from_any(u.get("allowed_departments")),
                )
                res["password"] = generated_password  # printed only if generated
                results.append(res)

            print(f"✅ Usuarios procesados: {len(results)}")
            for r in results:
                extra = ""
                if r.get("password"):
                    extra = f" | password generado: {r['password']}"
                print(
                    f" - {r['username']} | {'CREADO' if r['created'] else 'ACTUALIZADO'} | role={r['role']} | "
                    f"active={r['is_active']} | allowed_departments={r.get('allowed_departments')}{extra}"
                )
            return 0

        # Single user mode
        username = (args.username or "").strip()
        if not username:
            raise ValueError("--username is required (or use --users-file)")

        role = (args.role or "user").strip()
        is_active = not args.inactive

        existing = User.query.filter_by(username=username).first()

        password = args.password
        generated_password = None
        if not password and args.generate_password:
            generated_password = _generate_password()
            password = generated_password

        # Password handling rules:
        # - New user: password required (prompt if not provided)
        # - Existing user: password required only if --reset-password

        def _prompt_password(label: str) -> str:
            # getpass hides what is typed (no characters appear). It fails on some terminals.
            if not sys.stdin.isatty():
                raise ValueError(
                    "No hay terminal interactiva (stdin no es TTY). Usa --password o --generate-password."
                )
            try:
                print("(The password will not be shown while typing. Press Enter when done.)")
                return getpass(label)
            except Exception:
                # Fallback for terminals with poor getpass support (e.g. some embedded ones)
                return input(label.replace(":", " (visible): "))

        if not existing and not password:
            password = _prompt_password("Password: ")
        if existing and args.reset_password and not password:
            password = _prompt_password("New password: ")


        if existing and not args.reset_password:
            # Don't touch the password.
            password = None

        res = upsert_user(
            username=username,
            password=password,
            role=role,
            is_active=is_active,
            email=args.email,
            reset_password=args.reset_password,
            allowed_departments=_departments_from_any(args.allowed_departments),
        )

        print(
            f"✅ {res['username']} | {'CREADO' if res['created'] else 'ACTUALIZADO'} | role={res['role']} | "
            f"active={res['is_active']} | allowed_departments={res.get('allowed_departments')}"
        )
        if generated_password:
            print("🔑 Generated password (save it now, it will not be shown again):", generated_password)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        raise
